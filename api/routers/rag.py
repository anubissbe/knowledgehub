"""
RAG (Retrieval-Augmented Generation) API endpoints
Implements the production RAG system with LlamaIndex
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from ..services.rag.llamaindex_service import get_rag_service
from ..services.rag.documentation_scraper import get_scraper_service
from ..services.rag.contextual_enrichment import get_enrichment_service
from ..services.auth import get_current_user
from ..core.database import get_session
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
    stream: bool = Field(False, description="Enable streaming response")


class ScrapingJobRequest(BaseModel):
    """Request model for documentation scraping jobs"""
    site_name: str = Field(..., description="Documentation site to scrape")
    max_pages: int = Field(100, ge=1, le=500, description="Maximum pages to scrape")
    check_changes: bool = Field(True, description="Only scrape changed content")


class EnrichmentEstimateRequest(BaseModel):
    """Request model for enrichment cost estimation"""
    chunks: List[str] = Field(..., description="List of chunk texts")
    content_type: str = Field("documentation", description="Type of content")


# Initialize services
rag_service = get_rag_service()
scraper_service = get_scraper_service()
enrichment_service = get_enrichment_service()


@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_document(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Ingest a document into the RAG system with optional contextual enrichment
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
    Execute a RAG query with advanced features
    """
    try:
        # Execute query
        result = await rag_service.query(
            query_text=request.query,
            user_id=str(current_user.id),
            project_id=request.project_id,
            filters=request.filters,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            stream=request.stream
        )
        
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in result:
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            return result
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


@router.post("/scrape", response_model=Dict[str, Any])
async def start_scraping_job(
    request: ScrapingJobRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Start a documentation scraping job
    """
    try:
        # Check if site is valid
        if request.site_name not in scraper_service.DOCUMENTATION_SOURCES:
            available_sites = list(scraper_service.DOCUMENTATION_SOURCES.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown site: {request.site_name}. Available sites: {available_sites}"
            )
            
        # Check if job is already running
        job_status = await scraper_service.cache.get(f"scraping_job:{request.site_name}")
        if job_status and job_status.get("status") == "running":
            raise HTTPException(
                status_code=409,
                detail=f"Scraping job already running for {request.site_name}"
            )
            
        # Start job in background
        background_tasks.add_task(
            scraper_service.process_scraping_job,
            {
                "site_name": request.site_name,
                "max_pages": request.max_pages,
                "requested_by": str(current_user.id)
            }
        )
        
        # Set initial job status
        await scraper_service.cache.set(
            f"scraping_job:{request.site_name}",
            {
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "requested_by": str(current_user.id)
            },
            ttl=86400
        )
        
        return {
            "job_id": f"scraping_job:{request.site_name}",
            "status": "started",
            "site_name": request.site_name,
            "max_pages": request.max_pages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start scraping job: {str(e)}")


@router.get("/scrape/status/{site_name}", response_model=Dict[str, Any])
async def get_scraping_status(
    site_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a scraping job
    """
    job_status = await scraper_service.cache.get(f"scraping_job:{site_name}")
    
    if not job_status:
        raise HTTPException(status_code=404, detail=f"No job found for {site_name}")
        
    return job_status


@router.get("/scrape/stats", response_model=Dict[str, Any])
async def get_scraping_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get overall scraping statistics
    """
    stats = await scraper_service.get_scraping_stats()
    return stats


@router.post("/scrape/schedule", response_model=Dict[str, Any])
async def schedule_documentation_updates(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Schedule documentation updates for all configured sites
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
        
    background_tasks.add_task(scraper_service.schedule_documentation_updates)
    
    return {
        "status": "scheduled",
        "sites": list(scraper_service.DOCUMENTATION_SOURCES.keys()),
        "scheduled_at": datetime.utcnow().isoformat()
    }


@router.post("/enrich/estimate", response_model=Dict[str, Any])
async def estimate_enrichment_cost(
    request: EnrichmentEstimateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Estimate the cost of enriching document chunks
    """
    estimate = await enrichment_service.estimate_enrichment_cost(
        chunks=request.chunks,
        content_type=request.content_type
    )
    
    return estimate


@router.get("/index/stats", response_model=Dict[str, Any])
async def get_index_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get RAG index statistics
    """
    # Get cached stats first
    stats = await rag_service.cache.get("rag_index_stats")
    
    if not stats:
        # Update stats if not cached
        stats = await rag_service.update_index_stats()
        
    return stats


@router.post("/index/clear", response_model=Dict[str, Any])
async def clear_index(
    current_user: User = Depends(get_current_user)
):
    """
    Clear the RAG index (admin only)
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
        
    result = await rag_service.clear_index()
    
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to clear index"))
        
    return result


@router.get("/sources", response_model=List[Dict[str, Any]])
async def list_documentation_sources(
    current_user: User = Depends(get_current_user)
):
    """
    List available documentation sources for scraping
    """
    sources = []
    
    for name, config in scraper_service.DOCUMENTATION_SOURCES.items():
        # Get last update time
        last_update = await scraper_service.cache.get(f"last_doc_update:{name}")
        
        # Get job status
        job_status = await scraper_service.cache.get(f"scraping_job:{name}")
        
        sources.append({
            "name": name,
            "url": config["url"],
            "last_update": last_update,
            "job_status": job_status,
            "selector": config.get("selector", ""),
            "wait_for": config.get("wait_for", "")
        })
        
    return sources


@router.post("/test", response_model=Dict[str, Any])
async def test_rag_pipeline(
    current_user: User = Depends(get_current_user)
):
    """
    Test the RAG pipeline with sample data
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
        - Easy: Designed to be easy to use and learn
        - Short: Minimize code duplication
        - Robust: Get production-ready code with automatic documentation
        - Standards-based: Based on OpenAPI and JSON Schema
        
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
            "pipeline_working": True
        }
        
    except Exception as e:
        return {
            "test_status": "failed",
            "error": str(e),
            "pipeline_working": False
        }


# Health check endpoint
@router.get("/health", response_model=Dict[str, Any])
async def rag_health_check():
    """
    Check RAG system health
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "rag_service": "initialized" if rag_service.index is not None else "not_initialized",
            "scraper_service": "ready",
            "enrichment_service": "ready"
        }
    }
    
    # Check if index needs initialization
    if rag_service.index is None:
        try:
            await rag_service.initialize_index()
            health["services"]["rag_service"] = "initialized"
        except Exception as e:
            health["services"]["rag_service"] = f"error: {str(e)}"
            health["status"] = "degraded"
            
    return health