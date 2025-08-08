"""
GraphRAG API Router - Neo4j Enhanced RAG Endpoints
Dynamic Parallelism and Memory Bandwidth Optimization

This module provides REST API endpoints for GraphRAG functionality,
leveraging Neo4j knowledge graphs with vector RAG for enhanced retrieval.

Author: Charlotte Cools - Dynamic Parallelism Expert
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.graphrag_service import (
    GraphRAGService, GraphRAGStrategy, GraphRAGResult, 
    get_graphrag_service, EntityType, GraphMemoryConfig
)
from ..services.graph_aware_chunking import (
    ParallelGraphChunker, GraphChunkingStrategy, GraphChunk, ChunkingConfig
)
from ..dependencies import get_current_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/graphrag", tags=["GraphRAG"])


# Pydantic models for API
class DocumentInput(BaseModel):
    """Document input for indexing"""
    id: str = Field(..., description="Unique document ID")
    content: str = Field(..., description="Document content")
    title: Optional[str] = Field(None, description="Document title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    """Request to index documents with GraphRAG"""
    documents: List[DocumentInput] = Field(..., description="Documents to index")
    extract_entities: bool = Field(True, description="Whether to extract entities")
    build_relationships: bool = Field(True, description="Whether to build relationships")
    chunking_strategy: GraphChunkingStrategy = Field(
        GraphChunkingStrategy.SEMANTIC_GRAPH, 
        description="Graph-aware chunking strategy"
    )


class QueryRequest(BaseModel):
    """Request to query GraphRAG"""
    query: str = Field(..., description="Query text")
    strategy: GraphRAGStrategy = Field(
        GraphRAGStrategy.HYBRID_PARALLEL, 
        description="GraphRAG retrieval strategy"
    )
    max_results: int = Field(10, ge=1, le=100, description="Maximum results to return")
    include_reasoning: bool = Field(True, description="Include reasoning path")


class EntityExtractionRequest(BaseModel):
    """Request for entity extraction"""
    text: str = Field(..., description="Text to extract entities from")
    entity_types: Optional[List[EntityType]] = Field(None, description="Entity types to extract")


class GraphStatsResponse(BaseModel):
    """Graph statistics response"""
    nodes: int = Field(..., description="Number of nodes")
    relationships: int = Field(..., description="Number of relationships")
    entities: int = Field(..., description="Number of entities")
    documents: int = Field(..., description="Number of documents")
    communities: int = Field(..., description="Number of communities")


class MemoryStatsResponse(BaseModel):
    """Memory statistics response"""
    node_cache_size: int = Field(..., description="Node cache size")
    relationship_cache_size: int = Field(..., description="Relationship cache size")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    hit_ratio: float = Field(..., description="Cache hit ratio")
    max_memory_mb: int = Field(..., description="Maximum memory limit in MB")
    active_workers: int = Field(..., description="Active parallel workers")


class GraphRAGResultModel(BaseModel):
    """GraphRAG result model"""
    content: str = Field(..., description="Result content")
    score: float = Field(..., description="Combined relevance score")
    vector_score: float = Field(..., description="Vector similarity score")
    graph_score: float = Field(..., description="Graph relevance score")
    entities: List[str] = Field(..., description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(..., description="Entity relationships")
    reasoning_path: List[str] = Field(..., description="Reasoning path")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class ChunkingResultModel(BaseModel):
    """Graph chunking result model"""
    chunk_id: str = Field(..., description="Unique chunk ID")
    content: str = Field(..., description="Chunk content")
    entities: List[str] = Field(..., description="Entities in chunk")
    relationships: List[Dict[str, Any]] = Field(..., description="Entity relationships")
    coherence_score: float = Field(..., description="Semantic coherence score")
    entity_density: float = Field(..., description="Entity density in chunk")
    community_id: Optional[str] = Field(None, description="Graph community ID")
    metadata: Dict[str, Any] = Field(..., description="Chunk metadata")


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check for GraphRAG service"""
    try:
        service = await get_graphrag_service()
        neo4j_available = service.driver is not None
        
        return {
            "status": "healthy",
            "neo4j_connected": neo4j_available,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"GraphRAG health check failed: {e}")
        raise HTTPException(status_code=503, detail="GraphRAG service unavailable")


@router.post("/index", response_model=Dict[str, Any])
async def index_documents(
    request: IndexRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Index documents with GraphRAG including entity extraction and relationship building
    """
    try:
        service = await get_graphrag_service()
        
        # Convert Pydantic models to dictionaries
        documents = [doc.dict() for doc in request.documents]
        
        # Optional: Apply graph-aware chunking first
        if request.chunking_strategy != GraphChunkingStrategy.SEMANTIC_GRAPH:
            chunker = ParallelGraphChunker(
                config=ChunkingConfig(),
                neo4j_driver=service.driver
            )
            
            chunks = await chunker.chunk_documents_parallel(
                documents, 
                strategy=request.chunking_strategy
            )
            
            # Convert chunks back to documents for indexing
            documents = []
            for chunk in chunks:
                documents.append({
                    'id': chunk.chunk_id,
                    'content': chunk.content,
                    'title': f"Chunk from {chunk.metadata.get('doc_id', 'Unknown')}",
                    'metadata': {
                        **chunk.metadata,
                        'entities': chunk.entities,
                        'relationships': chunk.relationships,
                        'coherence_score': chunk.coherence_score,
                        'entity_density': chunk.entity_density
                    }
                })
        
        # Index with GraphRAG
        stats = await service.index_documents_with_graph(
            documents=documents,
            extract_entities=request.extract_entities,
            build_relationships=request.build_relationships
        )
        
        return {
            "status": "success",
            "message": f"Indexed {len(documents)} documents successfully",
            "statistics": stats,
            "chunking_strategy": request.chunking_strategy.value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/query", response_model=List[GraphRAGResultModel])
async def query_graphrag(
    request: QueryRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Query documents using GraphRAG with various retrieval strategies
    """
    try:
        service = await get_graphrag_service()
        
        results = await service.query_graphrag(
            query=request.query,
            strategy=request.strategy,
            max_results=request.max_results,
            include_reasoning=request.include_reasoning
        )
        
        # Convert GraphRAGResult objects to Pydantic models
        result_models = []
        for result in results:
            result_models.append(GraphRAGResultModel(
                content=result.content,
                score=result.score,
                vector_score=result.vector_score,
                graph_score=result.graph_score,
                entities=[e.entity if hasattr(e, 'entity') else str(e) for e in result.entities],
                relationships=result.relationships,
                reasoning_path=result.reasoning_path,
                metadata=result.metadata
            ))
        
        return result_models
        
    except Exception as e:
        logger.error(f"GraphRAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/extract-entities", response_model=List[Dict[str, Any]])
async def extract_entities(
    request: EntityExtractionRequest,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Extract entities from text using graph-aware entity recognition
    """
    try:
        service = await get_graphrag_service()
        
        # Extract entities using the processor
        entities = service.processor._extract_entities_from_text(request.text)
        
        # Convert EntityExtraction objects to dictionaries
        entity_results = []
        for entity in entities:
            if hasattr(entity, 'entity'):  # EntityExtraction object
                entity_results.append({
                    'entity': entity.entity,
                    'entity_type': entity.entity_type.value,
                    'confidence': entity.confidence,
                    'context': entity.context
                })
            else:  # Simple string entity
                entity_results.append({
                    'entity': str(entity),
                    'entity_type': 'UNKNOWN',
                    'confidence': 0.5,
                    'context': request.text[:200]
                })
        
        return entity_results
        
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.post("/chunk", response_model=List[ChunkingResultModel])
async def chunk_documents(
    documents: List[DocumentInput] = Body(...),
    strategy: GraphChunkingStrategy = Query(GraphChunkingStrategy.SEMANTIC_GRAPH),
    target_chunk_size: int = Query(512, ge=128, le=2048),
    max_chunk_size: int = Query(1024, ge=256, le=4096),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Apply graph-aware chunking to documents
    """
    try:
        service = await get_graphrag_service()
        
        # Create chunking configuration
        config = ChunkingConfig(
            target_chunk_size=target_chunk_size,
            max_chunk_size=max_chunk_size
        )
        
        # Create chunker
        chunker = ParallelGraphChunker(
            config=config,
            neo4j_driver=service.driver
        )
        
        # Convert documents to dictionaries
        doc_dicts = [doc.dict() for doc in documents]
        
        # Apply chunking
        chunks = await chunker.chunk_documents_parallel(doc_dicts, strategy)
        
        # Convert chunks to response models
        chunk_models = []
        for chunk in chunks:
            chunk_models.append(ChunkingResultModel(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                entities=chunk.entities,
                relationships=chunk.relationships,
                coherence_score=chunk.coherence_score,
                entity_density=chunk.entity_density,
                community_id=chunk.community_id,
                metadata=chunk.metadata
            ))
        
        return chunk_models
        
    except Exception as e:
        logger.error(f"Document chunking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@router.get("/graph-stats", response_model=GraphStatsResponse)
async def get_graph_statistics(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get Neo4j graph statistics
    """
    try:
        service = await get_graphrag_service()
        
        if not service.driver:
            raise HTTPException(status_code=503, detail="Neo4j not available")
        
        stats = {}
        
        with service.driver.session() as session:
            # Count nodes by type
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            node_counts = {}
            total_nodes = 0
            
            for record in result:
                labels = record['labels']
                count = record['count']
                total_nodes += count
                
                for label in labels:
                    if label not in node_counts:
                        node_counts[label] = 0
                    node_counts[label] += count
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as relationship_count")
            relationship_count = result.single()['relationship_count']
            
            # Count communities (simplified)
            result = session.run("""
                MATCH (e:Entity)
                WITH e, size((e)-[]->()) as degree
                WHERE degree > 0
                RETURN count(distinct e) as connected_entities
            """)
            communities = result.single()['connected_entities'] // 10  # Rough estimate
        
        return GraphStatsResponse(
            nodes=total_nodes,
            relationships=relationship_count,
            entities=node_counts.get('Entity', 0),
            documents=node_counts.get('Document', 0),
            communities=max(communities, 1)
        )
        
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")


@router.get("/memory-stats", response_model=MemoryStatsResponse)
async def get_memory_statistics(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get GraphRAG service memory and performance statistics
    """
    try:
        service = await get_graphrag_service()
        stats = service.get_memory_stats()
        
        return MemoryStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get memory statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Memory stats failed: {str(e)}")


@router.get("/strategies", response_model=Dict[str, List[str]])
async def get_available_strategies():
    """
    Get available GraphRAG and chunking strategies
    """
    return {
        "graphrag_strategies": [strategy.value for strategy in GraphRAGStrategy],
        "chunking_strategies": [strategy.value for strategy in GraphChunkingStrategy],
        "entity_types": [entity_type.value for entity_type in EntityType]
    }


@router.post("/benchmark", response_model=Dict[str, Any])
async def benchmark_graphrag(
    query: str = Body(..., embed=True),
    num_runs: int = Query(5, ge=1, le=20),
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Benchmark different GraphRAG strategies for performance comparison
    """
    try:
        service = await get_graphrag_service()
        
        strategies = [
            GraphRAGStrategy.VECTOR_FIRST,
            GraphRAGStrategy.GRAPH_FIRST,
            GraphRAGStrategy.HYBRID_PARALLEL,
            GraphRAGStrategy.ENTITY_CENTRIC
        ]
        
        benchmark_results = {}
        
        for strategy in strategies:
            strategy_times = []
            strategy_results = []
            
            for run in range(num_runs):
                start_time = datetime.now()
                
                results = await service.query_graphrag(
                    query=query,
                    strategy=strategy,
                    max_results=10,
                    include_reasoning=False  # Skip reasoning for performance
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds() * 1000  # ms
                
                strategy_times.append(execution_time)
                strategy_results.append(len(results))
            
            benchmark_results[strategy.value] = {
                'avg_time_ms': sum(strategy_times) / len(strategy_times),
                'min_time_ms': min(strategy_times),
                'max_time_ms': max(strategy_times),
                'avg_results': sum(strategy_results) / len(strategy_results),
                'runs': num_runs
            }
        
        # Memory stats
        memory_stats = service.get_memory_stats()
        
        return {
            'query': query,
            'benchmark_results': benchmark_results,
            'memory_stats': memory_stats,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"GraphRAG benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.delete("/clear-cache", response_model=Dict[str, str])
async def clear_cache(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Clear GraphRAG service caches to free memory
    """
    try:
        service = await get_graphrag_service()
        
        # Clear caches
        service.node_cache.clear()
        service.relationship_cache.clear()
        
        # Clear processor caches
        service.processor._clear_caches()
        
        return {
            'status': 'success',
            'message': 'All caches cleared successfully',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


@router.get("/config", response_model=Dict[str, Any])
async def get_graphrag_config(
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """
    Get current GraphRAG service configuration
    """
    try:
        service = await get_graphrag_service()
        
        return {
            'memory_config': {
                'max_memory_mb': service.config.max_memory_mb,
                'chunk_size_mb': service.config.chunk_size_mb,
                'prefetch_factor': service.config.prefetch_factor,
                'max_workers': service.config.max_workers,
                'batch_size': service.config.batch_size,
                'memory_threshold': service.config.memory_threshold,
                'max_depth': service.config.max_depth,
                'node_cache_size': service.config.node_cache_size,
                'relationship_cache_size': service.config.relationship_cache_size
            },
            'neo4j_connected': service.driver is not None,
            'rag_pipeline_initialized': service.rag_pipeline is not None,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=f"Config retrieval failed: {str(e)}")
