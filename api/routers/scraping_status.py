"""Scraping job status router"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional, Any
from uuid import UUID
import redis
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Redis connection for job queues
redis_client = redis.Redis(host='localhost', port=6381, decode_responses=True)


@router.get("/status", response_model=Dict[str, Any])
async def get_scraping_status():
    """Get current scraping job status for all sources"""
    try:
        # Get queue lengths
        queue_status = {
            "high": redis_client.llen("crawl_jobs:high"),
            "normal": redis_client.llen("crawl_jobs:normal"), 
            "low": redis_client.llen("crawl_jobs:low")
        }
        
        # Get currently processing job if any
        current_job = None
        
        # Check each queue for the first job (which would be processing)
        for priority in ["high", "normal", "low"]:
            queue = f"crawl_jobs:{priority}"
            job_data = redis_client.lindex(queue, 0)
            if job_data:
                try:
                    job = json.loads(job_data)
                    current_job = {
                        "source_id": job.get("source", {}).get("id"),
                        "source_name": job.get("source_name"),
                        "priority": priority,
                        "status": "processing"
                    }
                    break
                except:
                    pass
        
        return {
            "queue_status": queue_status,
            "total_pending": sum(queue_status.values()),
            "current_job": current_job
        }
        
    except Exception as e:
        logger.error(f"Error getting scraping status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scraping status")


@router.get("/source/{source_id}/status", response_model=Dict[str, Any])
async def get_source_scraping_status(source_id: UUID):
    """Get scraping status for a specific source"""
    try:
        source_id_str = str(source_id)
        
        # Check if this source is in any queue
        for priority in ["high", "normal", "low"]:
            queue = f"crawl_jobs:{priority}"
            queue_length = redis_client.llen(queue)
            
            for i in range(queue_length):
                job_data = redis_client.lindex(queue, i)
                if job_data:
                    try:
                        job = json.loads(job_data)
                        if job.get("source", {}).get("id") == source_id_str:
                            return {
                                "status": "processing" if i == 0 else "queued",
                                "priority": priority,
                                "position": i + 1,
                                "queue_length": queue_length
                            }
                    except:
                        continue
        
        # Not in any queue
        return {
            "status": "idle",
            "priority": None,
            "position": None,
            "queue_length": None
        }
        
    except Exception as e:
        logger.error(f"Error getting source scraping status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get source scraping status")


@router.get("/sources/status", response_model=Dict[str, Dict])
async def get_all_sources_scraping_status():
    """Get scraping status for all sources"""
    try:
        source_statuses = {}
        
        # Collect all jobs from all queues
        for priority in ["high", "normal", "low"]:
            queue = f"crawl_jobs:{priority}"
            queue_length = redis_client.llen(queue)
            
            for i in range(queue_length):
                job_data = redis_client.lindex(queue, i)
                if job_data:
                    try:
                        job = json.loads(job_data)
                        source_id = job.get("source", {}).get("id")
                        if source_id:
                            source_statuses[source_id] = {
                                "status": "processing" if i == 0 and priority == "high" else "queued",
                                "priority": priority,
                                "position": i + 1,
                                "source_name": job.get("source_name")
                            }
                    except:
                        continue
        
        return source_statuses
        
    except Exception as e:
        logger.error(f"Error getting all sources scraping status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sources scraping status")