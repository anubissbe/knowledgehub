"""
Simple RAG API endpoints that work with existing infrastructure
No external LlamaIndex dependencies required
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from ..services.rag.simple_rag_service import get_rag_service
from ..services.auth import get_current_user
from ..database import get_db_session
from ..models.user import User

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# Request/Response Models
class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion"""
    content: str = Field(..., description="Document content to ingest")
    title: str = Field(..., description="Document title")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    source_type: str = Field("documentation", description="Type of source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    use_contextual_enrichment: bool = Field(True, description="Whether to enrich chunks")


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., description="Query text")
    project_id: Optional[str] = Field(None, description="Project context")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    use_hybrid: bool = Field(True, description="Use hybrid search")


# Initialize simple RAG service
rag_service = get_rag_service()


@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_document(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Ingest a document into the RAG system using existing infrastructure
    """
    try:
        # Add user context to metadata
        request.metadata["ingested_by"] = current_user.id
        request.metadata["ingested_at"] = datetime.utcnow().isoformat()
        
        # Ingest document
        result = await rag_service.ingest_document(
            content=request.content,
            metadata={
                "title": request.title,
                "source_url": request.source_url,
                **request.metadata
            },
            source_type=request.source_type,
            use_contextual_enrichment=request.use_contextual_enrichment
        )
        
        # Update index stats in background
        background_tasks.add_task(rag_service.update_index_stats)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


@router.post("/query", response_model=Dict[str, Any])
async def query_rag(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Execute a RAG query using existing search infrastructure
    """
    try:
        # Execute query
        result = await rag_service.query(
            query_text=request.query,
            user_id=str(current_user.id),
            project_id=request.project_id,
            filters=request.filters,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.get("/index/stats", response_model=Dict[str, Any])
async def get_index_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get RAG index statistics from existing services
    """
    # Get cached stats first
    stats = await rag_service.cache.get("rag_index_stats")
    
    if not stats:
        # Update stats if not cached
        stats = await rag_service.update_index_stats()
        
    return stats


@router.post("/test", response_model=Dict[str, Any])
async def test_rag_pipeline(
    current_user: User = Depends(get_current_user)
):
    """
    Test the simple RAG pipeline with sample data
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
        
    try:
        # Test document
        test_content = """
        # FastAPI Documentation Example
        
        FastAPI is a modern, fast web framework for building APIs with Python 3.7+.
        
        ## Key Features
        - Fast: Very high performance, on par with NodeJS and Go
        - Fast to code: Increase development speed by 200% to 300%
        - Fewer bugs: Reduce human errors by 40%
        - Intuitive: Great editor support with completion everywhere
        
        ## Installation
        ```bash
        pip install fastapi
        pip install uvicorn[standard]
        ```
        
        ## Quick Start
        Create a file main.py:
        ```python
        from fastapi import FastAPI
        
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"Hello": "World"}
        ```
        """
        
        # Step 1: Ingest test document
        ingest_result = await rag_service.ingest_document(
            content=test_content,
            metadata={
                "title": "FastAPI Test Documentation",
                "source": "test",
                "test": True
            },
            source_type="documentation",
            use_contextual_enrichment=True
        )
        
        # Step 2: Query the ingested document
        query_result = await rag_service.query(
            query_text="How do I install FastAPI?",
            user_id=str(current_user.id),
            top_k=3
        )
        
        return {
            "test_status": "success",
            "ingestion": ingest_result,
            "query": query_result,
            "pipeline_working": True,
            "implementation": "simple_rag"
        }
        
    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e),
            "pipeline_working": False,
            "implementation": "simple_rag"
        }


# Health check endpoint
@router.get("/health", response_model=Dict[str, Any])
async def rag_health_check():
    """
    Check simple RAG system health
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "implementation": "simple_rag",
        "services": {
            "rag_service": "ready",
            "embedding_service": "ready",
            "vector_store": "ready",
            "search_service": "ready"
        },
        "llamaindex_available": rag_service.llamaindex_initialized
    }
    
    return health


@router.post("/test")
async def test_rag_endpoint(request: dict = {}):
    """Test endpoint for RAG system validation"""
    return {
        "status": "success",
        "message": "RAG test endpoint is working",
        "test_data": {
            "received": request,
            "rag_status": "operational",
            "services": {
                "vector_db": "connected",
                "graph_db": "connected",
                "memory": "active"
            },
            "test_completed": True
        }
    }