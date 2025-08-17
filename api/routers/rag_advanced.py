"""
Advanced RAG Router for KnowledgeHub
Exposes state-of-the-art RAG capabilities via REST API
"""

from fastapi import APIRouter, HTTPException, Depends, Body, Query, UploadFile, File
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import logging

from ..models import get_db
from ..services.rag_pipeline import (
    RAGConfig, RAGPipeline, ChunkingStrategy, RetrievalStrategy,
    Document
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag", tags=["rag-advanced"])

# Global RAG pipeline instance
rag_pipeline = None


class RAGQueryRequest(BaseModel):
    """Request model for RAG query"""
    query: str = Field(..., description="The query to process")
    retrieval_strategy: Optional[str] = Field(None, description="Retrieval strategy to use")
    top_k: Optional[int] = Field(10, description="Number of chunks to retrieve")
    enable_reranking: Optional[bool] = Field(True, description="Enable reranking")
    enable_hyde: Optional[bool] = Field(False, description="Enable HyDE")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class RAGIngestRequest(BaseModel):
    """Request model for document ingestion"""
    content: str = Field(..., description="Document content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")
    chunking_strategy: Optional[str] = Field("hierarchical", description="Chunking strategy")
    chunk_size: Optional[int] = Field(512, description="Target chunk size")
    chunk_overlap: Optional[int] = Field(128, description="Chunk overlap size")


class RAGConfigUpdate(BaseModel):
    """Request model for updating RAG configuration"""
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunking_strategy: Optional[str] = None
    top_k: Optional[int] = None
    retrieval_strategy: Optional[str] = None
    similarity_threshold: Optional[float] = None
    enable_reranking: Optional[bool] = None
    rerank_top_k: Optional[int] = None
    enable_hyde: Optional[bool] = None
    enable_graph_rag: Optional[bool] = None
    enable_self_correction: Optional[bool] = None


def get_rag_pipeline(db: Session = Depends(get_db)) -> RAGPipeline:
    """Get or create RAG pipeline instance"""
    global rag_pipeline
    if rag_pipeline is None:
        config = RAGConfig()
        rag_pipeline = RAGPipeline(config, db)
    return rag_pipeline


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "rag-advanced",
        "description": "Advanced RAG pipeline with state-of-the-art techniques",
        "features": [
            "Hierarchical chunking",
            "Hybrid retrieval",
            "Graph-enhanced RAG",
            "Self-correction",
            "HyDE support"
        ]
    }


@router.post("/query")
async def process_query(
    request: RAGQueryRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Process a query through the advanced RAG pipeline
    
    Features:
    - Multiple retrieval strategies (vector, hybrid, ensemble, graph)
    - Advanced reranking
    - HyDE (Hypothetical Document Embedding)
    - Self-correction mechanisms
    """
    try:
        pipeline = get_rag_pipeline(db)
        
        # Update configuration for this query if specified
        if request.retrieval_strategy:
            pipeline.config.retrieval_strategy = RetrievalStrategy(request.retrieval_strategy)
        if request.top_k:
            pipeline.config.top_k = request.top_k
        pipeline.config.enable_reranking = request.enable_reranking
        pipeline.config.enable_hyde = request.enable_hyde
        
        # Process query
        result = await pipeline.process_query(request.query, request.context)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        logger.error(f"RAG query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_document(
    request: RAGIngestRequest,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Ingest a document into the RAG system
    
    Features:
    - Multiple chunking strategies (semantic, hierarchical, proposition)
    - Automatic embedding generation
    - Metadata preservation
    """
    try:
        pipeline = get_rag_pipeline(db)
        
        # Update chunking configuration if specified
        if request.chunking_strategy:
            pipeline.config.chunking_strategy = ChunkingStrategy(request.chunking_strategy)
        if request.chunk_size:
            pipeline.config.chunk_size = request.chunk_size
        if request.chunk_overlap:
            pipeline.config.chunk_overlap = request.chunk_overlap
        
        # Ingest document
        document = await pipeline.ingest_document(request.content, request.metadata)
        
        return {
            "status": "success",
            "document_id": document.id,
            "chunks_created": len(document.chunks),
            "metadata": document.metadata
        }
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    chunking_strategy: str = Query("hierarchical"),
    chunk_size: int = Query(512),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Ingest a file into the RAG system
    
    Supports various file formats:
    - Text files (.txt, .md)
    - PDFs
    - Word documents
    - Code files
    """
    try:
        pipeline = get_rag_pipeline(db)
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8', errors='ignore')
        
        # Create metadata from file info
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content)
        }
        
        # Update configuration
        pipeline.config.chunking_strategy = ChunkingStrategy(chunking_strategy)
        pipeline.config.chunk_size = chunk_size
        
        # Ingest document
        document = await pipeline.ingest_document(content_str, metadata)
        
        return {
            "status": "success",
            "filename": file.filename,
            "document_id": document.id,
            "chunks_created": len(document.chunks),
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"File ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_configuration(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get current RAG configuration"""
    try:
        pipeline = get_rag_pipeline(db)
        config = pipeline.config
        
        return {
            "status": "success",
            "configuration": {
                "chunking": {
                    "strategy": config.chunking_strategy.value,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap
                },
                "retrieval": {
                    "strategy": config.retrieval_strategy.value,
                    "top_k": config.top_k,
                    "similarity_threshold": config.similarity_threshold
                },
                "reranking": {
                    "enabled": config.enable_reranking,
                    "top_k": config.rerank_top_k
                },
                "advanced_features": {
                    "hyde": config.enable_hyde,
                    "graph_rag": config.enable_graph_rag,
                    "self_correction": config.enable_self_correction
                },
                "performance": {
                    "caching": config.enable_caching,
                    "cache_ttl": config.cache_ttl,
                    "compression": config.enable_compression
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/config")
async def update_configuration(
    updates: RAGConfigUpdate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update RAG configuration"""
    try:
        pipeline = get_rag_pipeline(db)
        config = pipeline.config
        
        # Update configuration
        if updates.chunk_size is not None:
            config.chunk_size = updates.chunk_size
        if updates.chunk_overlap is not None:
            config.chunk_overlap = updates.chunk_overlap
        if updates.chunking_strategy is not None:
            config.chunking_strategy = ChunkingStrategy(updates.chunking_strategy)
        if updates.top_k is not None:
            config.top_k = updates.top_k
        if updates.retrieval_strategy is not None:
            config.retrieval_strategy = RetrievalStrategy(updates.retrieval_strategy)
        if updates.similarity_threshold is not None:
            config.similarity_threshold = updates.similarity_threshold
        if updates.enable_reranking is not None:
            config.enable_reranking = updates.enable_reranking
        if updates.rerank_top_k is not None:
            config.rerank_top_k = updates.rerank_top_k
        if updates.enable_hyde is not None:
            config.enable_hyde = updates.enable_hyde
        if updates.enable_graph_rag is not None:
            config.enable_graph_rag = updates.enable_graph_rag
        if updates.enable_self_correction is not None:
            config.enable_self_correction = updates.enable_self_correction
        
        return {
            "status": "success",
            "message": "Configuration updated",
            "configuration": {
                "chunking_strategy": config.chunking_strategy.value,
                "retrieval_strategy": config.retrieval_strategy.value,
                "top_k": config.top_k
            }
        }
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/strategies/chunking")
async def get_chunking_strategies() -> Dict[str, Any]:
    """Get available chunking strategies"""
    return {
        "status": "success",
        "strategies": [
            {
                "name": "semantic",
                "description": "Semantic-aware chunking based on sentence boundaries"
            },
            {
                "name": "sliding",
                "description": "Sliding window with configurable overlap"
            },
            {
                "name": "recursive",
                "description": "Recursive character splitting"
            },
            {
                "name": "proposition",
                "description": "Extract logical propositions"
            },
            {
                "name": "hierarchical",
                "description": "Multi-level hierarchical chunking (recommended)"
            },
            {
                "name": "adaptive",
                "description": "Context-aware adaptive sizing"
            }
        ]
    }


@router.get("/strategies/retrieval")
async def get_retrieval_strategies() -> Dict[str, Any]:
    """Get available retrieval strategies"""
    return {
        "status": "success",
        "strategies": [
            {
                "name": "vector",
                "description": "Pure vector similarity search"
            },
            {
                "name": "hybrid",
                "description": "Combine vector and keyword search (recommended)"
            },
            {
                "name": "ensemble",
                "description": "Multiple retrieval methods with voting"
            },
            {
                "name": "iterative",
                "description": "Progressive refinement"
            },
            {
                "name": "graph",
                "description": "Knowledge graph-enhanced retrieval"
            },
            {
                "name": "adaptive",
                "description": "Query-dependent strategy selection"
            }
        ]
    }


@router.get("/stats")
async def get_statistics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get RAG system statistics"""
    try:
        from sqlalchemy import text, func
        
        # Get document count
        doc_count = db.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        
        # Get chunk count
        chunk_count = db.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
        
        # Get average chunk size
        avg_chunk_size = db.execute(text(
            "SELECT AVG(LENGTH(content)) FROM chunks"
        )).scalar() or 0
        
        # Get documents with embeddings
        embedded_chunks = db.execute(text(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"
        )).scalar()
        
        return {
            "status": "success",
            "statistics": {
                "documents": doc_count or 0,
                "chunks": chunk_count or 0,
                "average_chunk_size": round(avg_chunk_size),
                "embedded_chunks": embedded_chunks or 0,
                "embedding_coverage": f"{(embedded_chunks / chunk_count * 100) if chunk_count > 0 else 0:.1f}%"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
async def reindex_documents(
    db: Session = Depends(get_db),
    chunking_strategy: str = Query("hierarchical")
) -> Dict[str, Any]:
    """
    Reindex all documents with new chunking strategy
    
    Warning: This operation can take a long time for large datasets
    """
    try:
        pipeline = get_rag_pipeline(db)
        pipeline.config.chunking_strategy = ChunkingStrategy(chunking_strategy)
        
        # Get all documents
        from sqlalchemy import text
        documents = db.execute(text("SELECT id, content, metadata FROM documents")).fetchall()
        
        total_chunks = 0
        for doc in documents:
            # Re-ingest document
            document = await pipeline.ingest_document(
                doc.content,
                doc.metadata
            )
            total_chunks += len(document.chunks)
        
        return {
            "status": "success",
            "documents_reindexed": len(documents),
            "total_chunks_created": total_chunks
        }
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a document and its chunks"""
    try:
        from sqlalchemy import text
        
        # Delete chunks
        db.execute(text("DELETE FROM chunks WHERE document_id = :doc_id"), {"doc_id": document_id})
        
        # Delete document
        result = db.execute(text("DELETE FROM documents WHERE id = :doc_id"), {"doc_id": document_id})
        db.commit()
        
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))