"""
LlamaIndex RAG Router for KnowledgeHub
Exposes LlamaIndex RAG orchestration via REST API
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query, UploadFile, File
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging

from ..models import get_db
from ..services.llamaindex_rag_service import (
    LlamaIndexRAGService, LlamaIndexConfig, LlamaIndexRAGStrategy,
    CompressionMethod, create_llamaindex_tables
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/llamaindex", tags=["llamaindex-rag"])

# Global service instance
llamaindex_service = None


class LlamaIndexQueryRequest(BaseModel):
    """Request model for LlamaIndex query"""
    query: str = Field(..., description="The query to process")
    index_id: str = Field(..., description="Index ID to query")
    strategy: Optional[str] = Field(None, description="RAG strategy override")
    top_k: Optional[int] = Field(10, description="Number of chunks to retrieve")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Chat history for conversational RAG")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class LlamaIndexCreateRequest(BaseModel):
    """Request model for creating a LlamaIndex"""
    documents: List[Dict[str, Any]] = Field(..., description="Documents to index")
    config: Optional[Dict[str, Any]] = Field(None, description="Index configuration")
    strategy: Optional[str] = Field("query_engine", description="RAG strategy")
    compression_enabled: Optional[bool] = Field(True, description="Enable compression")
    compression_method: Optional[str] = Field("truncated_svd", description="Compression method")
    compression_rank: Optional[int] = Field(128, description="Compression rank")


class LlamaIndexConfigUpdate(BaseModel):
    """Request model for updating LlamaIndex configuration"""
    strategy: Optional[str] = None
    compression_enabled: Optional[bool] = None
    compression_method: Optional[str] = None
    compression_rank: Optional[int] = None
    compression_ratio: Optional[float] = None
    similarity_top_k: Optional[int] = None
    response_mode: Optional[str] = None
    enable_streaming: Optional[bool] = None
    enable_citation: Optional[bool] = None


def get_llamaindex_service(db: Session = Depends(get_db)) -> LlamaIndexRAGService:
    """Get or create LlamaIndex service instance"""
    global llamaindex_service
    if llamaindex_service is None:
        # Ensure database tables exist
        try:
            create_llamaindex_tables(db)
        except Exception as e:
            logger.warning(f"Failed to create LlamaIndex tables: {e}")
        
        # Create service with default config
        config = LlamaIndexConfig()
        llamaindex_service = LlamaIndexRAGService(config, db)
    
    return llamaindex_service


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "llamaindex-rag",
        "description": "LlamaIndex RAG orchestration with low-rank factorization optimizations",
        "features": [
            "Query Engine",
            "Chat Engine", 
            "Sub-question decomposition",
            "Tree summarization",
            "Low-rank compression",
            "Mathematical optimizations"
        ]
    }


@router.post("/index/create")
async def create_index(
    request: LlamaIndexCreateRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a new LlamaIndex with mathematical optimizations
    
    Features:
    - Low-rank factorization compression
    - Multiple RAG strategies
    - Memory-efficient indexing
    - Performance optimizations
    """
    try:
        service = get_llamaindex_service(db)
        
        # Update service configuration if provided
        if request.config:
            for key, value in request.config.items():
                if hasattr(service.config, key):
                    setattr(service.config, key, value)
        
        # Set strategy
        if request.strategy:
            service.config.strategy = LlamaIndexRAGStrategy(request.strategy)
        
        # Set compression options
        service.config.enable_compression = request.compression_enabled
        if request.compression_method:
            service.config.compression_method = CompressionMethod(request.compression_method)
        if request.compression_rank:
            service.config.compression_rank = request.compression_rank
        
        # Create index
        index_id = await service.create_index_from_documents(request.documents)
        
        # Get index statistics
        stats = await service.get_index_statistics(index_id)
        
        return {
            "status": "success",
            "index_id": index_id,
            "documents_processed": len(request.documents),
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_index(
    request: LlamaIndexQueryRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Query a LlamaIndex using advanced RAG strategies
    
    Features:
    - Multiple query strategies (query_engine, chat_engine, sub_question, tree_summarize)
    - Compressed vector search
    - Mathematical optimizations
    - Conversational memory
    """
    try:
        service = get_llamaindex_service(db)
        
        # Override strategy if provided
        if request.strategy:
            service.config.strategy = LlamaIndexRAGStrategy(request.strategy)
        
        # Override top_k if provided
        if request.top_k:
            service.config.similarity_top_k = request.top_k
        
        # Query the index
        result = await service.query_index(
            request.index_id,
            request.query,
            chat_history=request.chat_history,
            context=request.context,
            top_k=request.top_k
        )
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/{index_id}/stats")
async def get_index_statistics(
    index_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get comprehensive statistics for an index"""
    try:
        service = get_llamaindex_service(db)
        stats = await service.get_index_statistics(index_id)
        
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/index/{index_id}/config")
async def update_index_configuration(
    index_id: str,
    updates: LlamaIndexConfigUpdate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update index configuration"""
    try:
        service = get_llamaindex_service(db)
        
        # Convert updates to dictionary
        update_dict = {}
        for field, value in updates.dict(exclude_unset=True).items():
            if value is not None:
                update_dict[field] = value
        
        result = await service.update_index_configuration(index_id, update_dict)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies")
async def get_available_strategies() -> Dict[str, Any]:
    """Get available RAG strategies"""
    return {
        "status": "success",
        "strategies": [
            {
                "name": "query_engine",
                "description": "Basic query engine for straightforward Q&A",
                "use_case": "Simple factual queries"
            },
            {
                "name": "chat_engine",
                "description": "Conversational RAG with memory",
                "use_case": "Multi-turn conversations, follow-up questions"
            },
            {
                "name": "sub_question",
                "description": "Decompose complex queries into sub-questions",
                "use_case": "Complex multi-part queries"
            },
            {
                "name": "tree_summarize",
                "description": "Hierarchical summarization of information",
                "use_case": "Summarizing large amounts of information"
            },
            {
                "name": "router_query",
                "description": "Route queries to specialized indexes",
                "use_case": "Multi-domain knowledge bases"
            },
            {
                "name": "fusion_retrieval",
                "description": "Combine multiple retrieval approaches",
                "use_case": "Maximum retrieval quality"
            }
        ]
    }


@router.get("/compression/methods")
async def get_compression_methods() -> Dict[str, Any]:
    """Get available compression methods"""
    return {
        "status": "success",
        "methods": [
            {
                "name": "truncated_svd",
                "description": "Truncated Singular Value Decomposition (recommended)",
                "memory_efficiency": "High",
                "query_speed": "Fast"
            },
            {
                "name": "sparse_projection",
                "description": "Sparse Random Projection",
                "memory_efficiency": "Very High",
                "query_speed": "Very Fast"
            },
            {
                "name": "pca",
                "description": "Principal Component Analysis",
                "memory_efficiency": "Medium",
                "query_speed": "Medium"
            }
        ]
    }


@router.post("/benchmark")
async def benchmark_compression(
    embeddings_shape: tuple = Body(..., description="Shape of embeddings matrix (rows, cols)"),
    compression_methods: List[str] = Body(["truncated_svd", "sparse_projection"]),
    compression_ranks: List[int] = Body([64, 128, 256]),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Benchmark different compression methods and ranks"""
    try:
        service = get_llamaindex_service(db)
        
        # Generate synthetic embeddings for benchmarking
        import numpy as np
        embeddings = np.random.randn(embeddings_shape[0], embeddings_shape[1]).astype(np.float32)
        
        results = []
        
        for method_name in compression_methods:
            for rank in compression_ranks:
                # Configure for this benchmark
                original_config = service.config
                service.config.compression_method = CompressionMethod(method_name)
                service.config.compression_rank = rank
                
                # Perform compression
                compressed_index = service.optimizer.compress_embeddings(embeddings)
                
                # Calculate metrics
                memory_stats = service.optimizer.estimate_memory_savings(
                    embeddings_shape, compressed_index
                )
                
                results.append({
                    "method": method_name,
                    "rank": rank,
                    "compression_ratio": compressed_index.compression_ratio,
                    "memory_savings_mb": memory_stats["original_memory_mb"] - memory_stats["compressed_memory_mb"],
                    "memory_savings_percent": memory_stats["memory_savings_ratio"] * 100,
                    "original_shape": embeddings_shape,
                    "compressed_dimensions": compressed_index.u_matrix.shape
                })
                
                # Restore original config
                service.config = original_config
        
        return {
            "status": "success",
            "benchmark_results": results,
            "recommendations": {
                "best_memory_efficiency": min(results, key=lambda x: x["memory_savings_mb"]),
                "best_compression_ratio": max(results, key=lambda x: x["compression_ratio"])
            }
        }
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/index/{index_id}")
async def delete_index(
    index_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete an index and its associated data"""
    try:
        # Delete from database
        from sqlalchemy import text
        
        # Delete chunks
        db.execute(
            text("DELETE FROM index_chunks WHERE index_id = :index_id"),
            {"index_id": index_id}
        )
        
        # Delete index
        result = db.execute(
            text("DELETE FROM llamaindex_indexes WHERE id = :index_id"),
            {"index_id": index_id}
        )
        
        db.commit()
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Index not found")
        
        # Remove from memory if present
        service = get_llamaindex_service(db)
        if index_id in service.compressed_indexes:
            del service.compressed_indexes[index_id]
        
        return {
            "status": "success",
            "message": f"Index {index_id} deleted successfully"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indexes")
async def list_indexes(
    limit: int = Query(10, description="Maximum number of indexes to return"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """List all available indexes"""
    try:
        from sqlalchemy import text
        
        indexes = db.execute(
            text("""
                SELECT id, strategy, created_at, 
                       (SELECT COUNT(*) FROM index_chunks WHERE index_id = li.id) as chunk_count
                FROM llamaindex_indexes li
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            {"limit": limit, "offset": offset}
        ).fetchall()
        
        total_count = db.execute(
            text("SELECT COUNT(*) FROM llamaindex_indexes")
        ).scalar()
        
        return {
            "status": "success",
            "indexes": [
                {
                    "id": idx.id,
                    "strategy": idx.strategy,
                    "chunk_count": idx.chunk_count,
                    "created_at": idx.created_at.isoformat()
                }
                for idx in indexes
            ],
            "total_count": total_count,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")
        raise HTTPException(status_code=500, detail=str(e))
