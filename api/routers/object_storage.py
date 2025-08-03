"""
Object Storage Router

Provides REST API endpoints for object storage operations using MinIO.
"""

import os
from typing import Optional, List
from datetime import timedelta
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import logging

from ..services.object_storage_service import object_storage_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/storage", tags=["storage"])


@router.get("/health")
async def health_check():
    """Check object storage health and connectivity"""
    stats = object_storage_service.get_storage_stats()
    return {
        "status": "healthy" if stats["connected"] else "unhealthy",
        "stats": stats
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    bucket: str = Form("documents"),
    path: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Upload a file to object storage"""
    try:
        # Construct object name
        if path:
            object_name = f"{path}/{file.filename}"
        else:
            object_name = file.filename
        
        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            import json
            try:
                file_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata format, ignoring")
        
        # Add file info to metadata
        file_metadata["original_filename"] = file.filename
        file_metadata["content_type"] = file.content_type
        
        # Upload file
        result = object_storage_service.upload_file(
            bucket=bucket,
            object_name=object_name,
            file_data=file.file,
            content_type=file.content_type or "application/octet-stream",
            metadata=file_metadata
        )
        
        if result:
            return {
                "success": True,
                "file": result
            }
        else:
            # Check if bucket exists to provide better error message
            if object_storage_service.client and not object_storage_service.client.bucket_exists(bucket):
                raise HTTPException(status_code=400, detail=f"Bucket '{bucket}' does not exist. Available buckets: attachments, backups, documents, embeddings, temp")
            else:
                raise HTTPException(status_code=500, detail="Failed to upload file to storage")
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{bucket}/{object_name:path}")
async def download_file(bucket: str, object_name: str):
    """Download a file from object storage"""
    try:
        # Get file data
        file_data = object_storage_service.download_file(bucket, object_name)
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file info for content type
        file_info = object_storage_service.get_file_info(bucket, object_name)
        content_type = "application/octet-stream"
        if file_info and "content_type" in file_info:
            content_type = file_info["content_type"]
        
        # Return file as streaming response
        file_data.seek(0)
        return StreamingResponse(
            file_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(object_name)}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{bucket}/{object_name:path}")
async def get_file_info(bucket: str, object_name: str):
    """Get information about a stored file"""
    try:
        info = object_storage_service.get_file_info(bucket, object_name)
        if not info:
            raise HTTPException(status_code=404, detail="File not found")
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{bucket}/{object_name:path}")
async def delete_file(bucket: str, object_name: str):
    """Delete a file from object storage"""
    try:
        success = object_storage_service.delete_file(bucket, object_name)
        if success:
            return {"success": True, "message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found or deletion failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list/{bucket}")
async def list_files(
    bucket: str,
    prefix: Optional[str] = Query(None),
    recursive: bool = Query(False)
):
    """List files in a bucket"""
    try:
        files = object_storage_service.list_files(
            bucket=bucket,
            prefix=prefix,
            recursive=recursive
        )
        return {
            "bucket": bucket,
            "prefix": prefix,
            "recursive": recursive,
            "count": len(files),
            "files": files
        }
        
    except Exception as e:
        logger.error(f"List error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/presigned-url")
async def create_presigned_url(
    bucket: str = Form(...),
    object_name: str = Form(...),
    expires_minutes: int = Form(60)
):
    """Generate a presigned URL for temporary file access"""
    try:
        expires = timedelta(minutes=expires_minutes)
        url = object_storage_service.create_presigned_url(
            bucket=bucket,
            object_name=object_name,
            expires=expires
        )
        
        if url:
            return {
                "success": True,
                "url": url,
                "expires_in": expires_minutes * 60  # seconds
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to generate URL")
            
    except Exception as e:
        logger.error(f"Presigned URL error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bucket/{bucket}/versioning")
async def enable_bucket_versioning(bucket: str):
    """Enable versioning for a bucket"""
    try:
        success = object_storage_service.enable_versioning(bucket)
        if success:
            return {
                "success": True,
                "message": f"Versioning enabled for bucket: {bucket}"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to enable versioning")
            
    except Exception as e:
        logger.error(f"Versioning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_storage_statistics():
    """Get overall storage statistics"""
    try:
        stats = object_storage_service.get_storage_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))