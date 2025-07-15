"""Knowledge sources router"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import List, Optional
from uuid import UUID
from datetime import datetime
import logging

from ..dependencies import get_source_service, get_job_service
from ..schemas.source import SourceCreate, SourceUpdate, SourceResponse, SourceListResponse
from ..schemas.job import JobResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=JobResponse, status_code=202)
async def create_source(
    source: SourceCreate,
    background_tasks: BackgroundTasks,
    source_service=Depends(get_source_service),
    job_service=Depends(get_job_service)
):
    """
    Add a new knowledge source and start scraping.
    
    Returns a job ID while scraping happens asynchronously.
    """
    try:
        logger.info(f"Received source creation request: name={source.name}, url={source.url}, type={getattr(source, 'type', 'None')}, refresh_interval={getattr(source, 'refresh_interval', 'None')}")
        # Check if source already exists
        existing = await source_service.get_by_url(source.url)
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Source with this URL already exists"
            )
        
        # Create source with config
        config = source.to_db_config()
        logger.info(f"Creating source with config: {config}")
        
        # Generate name from URL if not provided
        name = source.name or str(source.url).replace("https://", "").replace("http://", "").replace("www.", "").strip("/")
        
        source_data = {
            "name": name,
            "url": str(source.url),
            "config": config
        }
        db_source = await source_service.create_from_dict(source_data)
        
        # Create scraping job
        job = await job_service.create_scraping_job(
            source_id=db_source.id,
            url=source.url
        )
        
        # Queue scraping task
        background_tasks.add_task(
            job_service.queue_scraping_job,
            job.id,
            db_source.id,
            source.url
        )
        
        return JobResponse(
            job_id=str(job.id),
            message=f"Scraping job created for {name}",
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=SourceListResponse)
async def list_sources(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    source_service=Depends(get_source_service)
):
    """Get list of all knowledge sources"""
    try:
        sources = await source_service.list_sources(
            skip=skip,
            limit=limit,
            status=status
        )
        
        total = await source_service.count_sources(status=status)
        
        # Convert sources using the new schema method
        source_responses = [SourceResponse.from_db_model(source) for source in sources]
        
        return SourceListResponse(
            sources=source_responses,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Error listing sources: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: UUID,
    source_service=Depends(get_source_service)
):
    """Get detailed information about a knowledge source"""
    try:
        source = await source_service.get_by_id(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return SourceResponse.from_db_model(source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: UUID,
    update: SourceUpdate,
    source_service=Depends(get_source_service)
):
    """Update knowledge source properties"""
    try:
        source = await source_service.update(source_id, update)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return SourceResponse.from_db_model(source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{source_id}", response_model=SourceResponse)
async def patch_source(
    source_id: UUID,
    update: SourceUpdate,
    source_service=Depends(get_source_service)
):
    """Partially update knowledge source properties (PATCH semantics)"""
    try:
        source = await source_service.update(source_id, update)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        return SourceResponse.from_db_model(source)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error patching source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{source_id}", response_model=JobResponse, status_code=202)
async def delete_source(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    source_service=Depends(get_source_service),
    job_service=Depends(get_job_service)
):
    """Delete a knowledge source and all its indexed content"""
    try:
        source = await source_service.get_by_id(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Create deletion job
        job = await job_service.create_deletion_job(source_id)
        
        # Queue deletion task
        background_tasks.add_task(
            source_service.delete_source_data,
            source_id
        )
        
        return JobResponse(
            job_id=str(job.id),
            message=f"Deletion job created for {source.name}",
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{source_id}/rescrape", response_model=JobResponse, status_code=202)
async def rescrape_source(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    source_service=Depends(get_source_service),
    job_service=Depends(get_job_service)
):
    """Trigger a new scraping job for an existing source"""
    try:
        source = await source_service.get_by_id(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Create rescraping job
        job = await job_service.create_scraping_job(
            source_id=source_id,
            url=source.url
        )
        
        # Queue scraping task
        background_tasks.add_task(
            job_service.queue_scraping_job,
            job.id,
            source_id,
            source.url
        )
        
        return JobResponse(
            job_id=str(job.id),
            message=f"Rescraping job created for {source.name}",
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rescraping source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{source_id}/refresh", response_model=JobResponse, status_code=202)
async def refresh_source(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    source_service=Depends(get_source_service),
    job_service=Depends(get_job_service)
):
    """Refresh an existing source by triggering a new scraping job (alias for rescrape)"""
    try:
        source = await source_service.get_by_id(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        # Create rescraping job
        job = await job_service.create_scraping_job(
            source_id=source_id,
            url=source.url
        )
        
        # Queue scraping task
        background_tasks.add_task(
            job_service.queue_scraping_job,
            job.id,
            source_id,
            source.url
        )
        
        return JobResponse(
            job_id=str(job.id),
            message=f"Refresh job created for {source.name}",
            status="queued"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing source: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")