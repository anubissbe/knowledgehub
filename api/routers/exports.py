"""
Export API Router

Provides comprehensive data export functionality for KnowledgeHub.
Supports multiple export formats and data types with filtering and pagination.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from ..models import get_db
from ..services.export_service import (
    ExportService, 
    ExportRequest, 
    ExportResult, 
    ExportFormat, 
    ExportType,
    get_export_service
)
from ..services.auth import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exports", tags=["exports"])


@router.post("/create", response_model=ExportResult)
async def create_export(
    export_request: ExportRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Create a new data export
    
    Creates an export job for the specified data type and format.
    Supports various filters and output formats.
    """
    try:
        export_service = get_export_service(db)
        
        # Validate export request
        if export_request.limit and export_request.limit > 10000:
            raise HTTPException(
                status_code=400,
                detail="Export limit cannot exceed 10,000 records"
            )
        
        # Create export
        result = await export_service.export_data(export_request)
        
        # Schedule cleanup in background
        background_tasks.add_task(
            export_service.cleanup_expired_exports
        )
        
        logger.info(f"Export created: {result.export_id} for user {current_user.get('id', 'unknown')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Export creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export creation failed: {str(e)}")


@router.get("/{export_id}/download")
async def download_export(
    export_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Download an export file
    
    Downloads the generated export file by export ID.
    Files are available for 24 hours after creation.
    """
    try:
        export_service = get_export_service(db)
        file_path = export_service.get_export_file(export_id)
        
        if not file_path:
            raise HTTPException(
                status_code=404,
                detail="Export file not found or expired"
            )
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="Export file no longer available"
            )
        
        # Determine filename and media type
        filename = os.path.basename(file_path)
        
        # Get media type based on file extension
        media_type_map = {
            '.json': 'application/json',
            '.csv': 'text/csv',
            '.pdf': 'application/pdf',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xml': 'application/xml',
            '.zip': 'application/zip'
        }
        
        file_ext = os.path.splitext(filename)[1].lower()
        media_type = media_type_map.get(file_ext, 'application/octet-stream')
        
        logger.info(f"Export downloaded: {export_id} by user {current_user.get('id', 'unknown')}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Export-ID": export_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/formats")
async def get_export_formats():
    """
    Get available export formats
    
    Returns all supported export formats and their descriptions.
    """
    formats = {
        ExportFormat.JSON.value: {
            "name": "JSON",
            "description": "JavaScript Object Notation - structured data format",
            "file_extension": ".json",
            "mime_type": "application/json",
            "supports_complex_data": True
        },
        ExportFormat.CSV.value: {
            "name": "CSV", 
            "description": "Comma-Separated Values - spreadsheet compatible",
            "file_extension": ".csv",
            "mime_type": "text/csv",
            "supports_complex_data": False
        },
        ExportFormat.PDF.value: {
            "name": "PDF",
            "description": "Portable Document Format - printable report format",
            "file_extension": ".pdf", 
            "mime_type": "application/pdf",
            "supports_complex_data": False
        },
        ExportFormat.EXCEL.value: {
            "name": "Excel",
            "description": "Microsoft Excel format - advanced spreadsheet",
            "file_extension": ".xlsx",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "supports_complex_data": False
        },
        ExportFormat.XML.value: {
            "name": "XML",
            "description": "Extensible Markup Language - structured markup",
            "file_extension": ".xml",
            "mime_type": "application/xml", 
            "supports_complex_data": True
        },
        ExportFormat.ZIP.value: {
            "name": "ZIP Archive",
            "description": "Compressed archive with multiple formats",
            "file_extension": ".zip",
            "mime_type": "application/zip",
            "supports_complex_data": True
        }
    }
    
    return {"formats": formats}


@router.get("/types")
async def get_export_types():
    """
    Get available export data types
    
    Returns all supported data types that can be exported.
    """
    types = {
        ExportType.SOURCES.value: {
            "name": "Knowledge Sources",
            "description": "Documentation sources and their configurations",
            "estimated_records": "Variable",
            "common_filters": ["status", "source_type", "created_after"]
        },
        ExportType.DOCUMENTS.value: {
            "name": "Documents",
            "description": "Processed documents from knowledge sources",
            "estimated_records": "Variable",
            "common_filters": ["source_id", "title", "content_type"]
        },
        ExportType.CHUNKS.value: {
            "name": "Text Chunks",
            "description": "Processed text chunks for search and analysis",
            "estimated_records": "High volume",
            "common_filters": ["document_id", "min_size", "max_size"]
        },
        ExportType.JOBS.value: {
            "name": "Processing Jobs",
            "description": "Background job processing history",
            "estimated_records": "Medium",
            "common_filters": ["status", "job_type", "source_id"]
        },
        ExportType.MEMORIES.value: {
            "name": "Memory Items",
            "description": "AI conversation memory and context",
            "estimated_records": "Variable",
            "common_filters": ["memory_type", "importance", "session_id"]
        },
        ExportType.THREAT_MODELS.value: {
            "name": "Threat Models",
            "description": "Security threat modeling diagrams and data",
            "estimated_records": "Low",
            "common_filters": ["status", "project_id"]
        },
        ExportType.ANALYTICS.value: {
            "name": "Analytics Data",
            "description": "System analytics and usage metrics",
            "estimated_records": "Summary",
            "common_filters": ["date_range", "metric_type"]
        },
        ExportType.SYSTEM_DATA.value: {
            "name": "System Information",
            "description": "System configuration and health data",
            "estimated_records": "Summary",
            "common_filters": ["component"]
        },
        ExportType.ALL.value: {
            "name": "Complete Export",
            "description": "All system data in a structured format",
            "estimated_records": "High volume",
            "common_filters": ["limit", "include_metadata"]
        }
    }
    
    return {"types": types}


@router.get("/quick-exports")
async def get_quick_export_options(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Get quick export options with record counts
    
    Provides predefined export options with current record counts.
    """
    from ..models.knowledge_source import KnowledgeSource
    from ..models.document import Document
    from ..models.chunk import Chunk
    from ..models.job import Job
    from ..models.memory import MemoryItem
    
    try:
        quick_options = [
            {
                "id": "recent_sources",
                "name": "Recent Knowledge Sources",
                "description": "Sources added in the last 30 days",
                "export_type": ExportType.SOURCES.value,
                "format": ExportFormat.CSV.value,
                "record_count": db.query(KnowledgeSource).count(),
                "filters": {"created_after": "30_days"}
            },
            {
                "id": "all_documents",
                "name": "All Documents",
                "description": "Complete document library export",
                "export_type": ExportType.DOCUMENTS.value,
                "format": ExportFormat.JSON.value,
                "record_count": db.query(Document).count(),
                "filters": {}
            },
            {
                "id": "failed_jobs",
                "name": "Failed Jobs",
                "description": "Jobs that ended with errors",
                "export_type": ExportType.JOBS.value,
                "format": ExportFormat.CSV.value,
                "record_count": db.query(Job).filter(Job.status == "failed").count(),
                "filters": {"status": "failed"}
            },
            {
                "id": "system_summary",
                "name": "System Summary",
                "description": "Complete system data summary",
                "export_type": ExportType.ALL.value,
                "format": ExportFormat.ZIP.value,
                "record_count": "Multiple",
                "filters": {"include_metadata": True}
            },
            {
                "id": "analytics_report",
                "name": "Analytics Report",
                "description": "System analytics and metrics",
                "export_type": ExportType.ANALYTICS.value,
                "format": ExportFormat.PDF.value,
                "record_count": "Summary",
                "filters": {}
            }
        ]
        
        return {"quick_exports": quick_options}
        
    except Exception as e:
        logger.error(f"Failed to get quick export options: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve export options")


@router.post("/quick/{option_id}")
async def create_quick_export(
    option_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Create export using quick option
    
    Creates an export using one of the predefined quick export options.
    """
    quick_options = {
        "recent_sources": ExportRequest(
            export_type=ExportType.SOURCES,
            export_format=ExportFormat.CSV,
            filters={"created_after": "2024-06-01"},  # Last 30 days
            include_metadata=True
        ),
        "all_documents": ExportRequest(
            export_type=ExportType.DOCUMENTS,
            export_format=ExportFormat.JSON,
            include_metadata=True
        ),
        "failed_jobs": ExportRequest(
            export_type=ExportType.JOBS,
            export_format=ExportFormat.CSV,
            filters={"status": "failed"},
            include_metadata=True
        ),
        "system_summary": ExportRequest(
            export_type=ExportType.ALL,
            export_format=ExportFormat.ZIP,
            include_metadata=True,
            include_statistics=True
        ),
        "analytics_report": ExportRequest(
            export_type=ExportType.ANALYTICS,
            export_format=ExportFormat.PDF,
            include_metadata=True,
            include_statistics=True
        )
    }
    
    if option_id not in quick_options:
        raise HTTPException(
            status_code=404,
            detail=f"Quick export option '{option_id}' not found"
        )
    
    try:
        export_service = get_export_service(db)
        export_request = quick_options[option_id]
        
        result = await export_service.export_data(export_request)
        
        # Schedule cleanup
        background_tasks.add_task(export_service.cleanup_expired_exports)
        
        logger.info(f"Quick export '{option_id}' created: {result.export_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Quick export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick export failed: {str(e)}")


@router.delete("/{export_id}")
async def delete_export(
    export_id: str,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Delete an export file
    
    Manually delete an export file before it expires.
    """
    try:
        export_service = get_export_service(db)
        file_path = export_service.get_export_file(export_id)
        
        if not file_path:
            raise HTTPException(
                status_code=404,
                detail="Export file not found"
            )
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Export deleted: {export_id} by user {current_user.get('id', 'unknown')}")
            
            return {"message": f"Export {export_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail="Export file not found on disk"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@router.post("/cleanup")
async def cleanup_exports(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Manually trigger export cleanup
    
    Removes all expired export files from the system.
    """
    try:
        export_service = get_export_service(db)
        
        # Run cleanup in background
        background_tasks.add_task(export_service.cleanup_expired_exports)
        
        logger.info(f"Export cleanup triggered by user {current_user.get('id', 'unknown')}")
        
        return {"message": "Export cleanup scheduled"}
        
    except Exception as e:
        logger.error(f"Export cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/stats")
async def get_export_stats(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Get export system statistics
    
    Returns statistics about available data and export system usage.
    """
    from ..models.knowledge_source import KnowledgeSource
    from ..models.document import Document
    from ..models.chunk import Chunk
    from ..models.job import Job
    from ..models.memory import MemoryItem
    
    try:
        export_service = get_export_service(db)
        
        # Count available records
        stats = {
            "available_records": {
                "sources": db.query(KnowledgeSource).count(),
                "documents": db.query(Document).count(),
                "chunks": db.query(Chunk).count(),
                "jobs": db.query(Job).count(),
                "memories": db.query(MemoryItem).count()
            },
            "export_system": {
                "temp_directory": export_service.temp_dir,
                "supported_formats": list(ExportFormat),
                "supported_types": list(ExportType),
                "max_records_per_export": 10000
            },
            "current_exports": {
                "active_files": len([f for f in os.listdir(export_service.temp_dir) 
                                   if os.path.isfile(os.path.join(export_service.temp_dir, f))]),
                "total_size_mb": sum(
                    os.path.getsize(os.path.join(export_service.temp_dir, f))
                    for f in os.listdir(export_service.temp_dir)
                    if os.path.isfile(os.path.join(export_service.temp_dir, f))
                ) / (1024 * 1024)
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get export stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve export statistics")


# Export templates for common use cases
@router.get("/templates")
async def get_export_templates():
    """
    Get export templates
    
    Returns predefined export templates for common use cases.
    """
    templates = {
        "data_backup": {
            "name": "Complete Data Backup",
            "description": "Full backup of all system data",
            "request": {
                "export_type": ExportType.ALL.value,
                "export_format": ExportFormat.ZIP.value,
                "include_metadata": True,
                "include_statistics": True
            }
        },
        "content_audit": {
            "name": "Content Audit Report",
            "description": "Sources and documents for content review",
            "request": {
                "export_type": ExportType.SOURCES.value,
                "export_format": ExportFormat.EXCEL.value,
                "include_metadata": True,
                "fields": ["id", "name", "url", "status", "documents_count", "last_crawled"]
            }
        },
        "processing_report": {
            "name": "Processing Status Report",
            "description": "Job processing history and status",
            "request": {
                "export_type": ExportType.JOBS.value,
                "export_format": ExportFormat.PDF.value,
                "include_metadata": True,
                "sort_by": "created_at",
                "sort_order": "desc"
            }
        },
        "memory_export": {
            "name": "AI Memory Export",
            "description": "AI conversation memory and context data",
            "request": {
                "export_type": ExportType.MEMORIES.value,
                "export_format": ExportFormat.JSON.value,
                "include_metadata": True,
                "filters": {"importance": 0.5}
            }
        }
    }
    
    return {"templates": templates}


@router.post("/templates/{template_id}")
async def create_export_from_template(
    template_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Create export using template
    
    Creates an export using one of the predefined templates.
    """
    templates = {
        "data_backup": ExportRequest(
            export_type=ExportType.ALL,
            export_format=ExportFormat.ZIP,
            include_metadata=True,
            include_statistics=True
        ),
        "content_audit": ExportRequest(
            export_type=ExportType.SOURCES,
            export_format=ExportFormat.EXCEL,
            include_metadata=True,
            fields=["id", "name", "url", "status", "documents_count", "last_crawled"]
        ),
        "processing_report": ExportRequest(
            export_type=ExportType.JOBS,
            export_format=ExportFormat.PDF,
            include_metadata=True,
            sort_by="created_at",
            sort_order="desc"
        ),
        "memory_export": ExportRequest(
            export_type=ExportType.MEMORIES,
            export_format=ExportFormat.JSON,
            include_metadata=True,
            filters={"importance": 0.5}
        )
    }
    
    if template_id not in templates:
        raise HTTPException(
            status_code=404,
            detail=f"Export template '{template_id}' not found"
        )
    
    try:
        export_service = get_export_service(db)
        export_request = templates[template_id]
        
        result = await export_service.export_data(export_request)
        
        # Schedule cleanup
        background_tasks.add_task(export_service.cleanup_expired_exports)
        
        logger.info(f"Template export '{template_id}' created: {result.export_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Template export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template export failed: {str(e)}")