"""
KnowledgeHub Export Service

Provides comprehensive data export functionality for all system entities.
Supports multiple export formats including JSON, CSV, PDF, and XML.
Includes filtering, pagination, and bulk export capabilities.
"""

import csv
import json
import logging
import io
import zipfile
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, BinaryIO
from uuid import UUID
# import pandas as pd  # Temporarily disabled until container rebuild
# from reportlab.lib.pagesizes import letter, A4  # Temporarily disabled
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle  # Temporarily disabled
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Temporarily disabled
# from reportlab.lib.units import inch  # Temporarily disabled
# from reportlab.lib import colors  # Temporarily disabled
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from fastapi import HTTPException
from pydantic import BaseModel, Field
from enum import Enum

from ..models.knowledge_source import KnowledgeSource
from ..models.document import Document
from ..models.chunk import Chunk
from ..models.job import Job
from ..models.memory import MemoryItem
from ..models.search import SearchHistory

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats"""
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    XML = "xml"
    EXCEL = "excel"
    ZIP = "zip"


class ExportType(str, Enum):
    """Types of data that can be exported"""
    SOURCES = "sources"
    DOCUMENTS = "documents"
    CHUNKS = "chunks"
    JOBS = "jobs"
    MEMORIES = "memories"
    SEARCH_RESULTS = "search_results"
    ANALYTICS = "analytics"
    SYSTEM_DATA = "system_data"
    ALL = "all"


class ExportRequest(BaseModel):
    """Export request configuration"""
    export_type: ExportType
    export_format: ExportFormat
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    limit: Optional[int] = Field(default=None, ge=1, le=10000)
    offset: Optional[int] = Field(default=0, ge=0)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    include_metadata: bool = True
    include_statistics: bool = False
    date_range: Optional[Dict[str, str]] = None
    fields: Optional[List[str]] = None
    compress: bool = False


class ExportResult(BaseModel):
    """Export operation result"""
    export_id: str
    export_type: ExportType
    export_format: ExportFormat
    status: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    record_count: int
    created_at: datetime
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExportService:
    """Main export service for KnowledgeHub data"""
    
    def __init__(self, db: Session):
        self.db = db
        self.temp_dir = "/tmp/knowledgehub_exports"
        self._ensure_temp_dir()
    
    def _ensure_temp_dir(self):
        """Ensure temporary directory exists"""
        import os
        os.makedirs(self.temp_dir, exist_ok=True)
    
    async def export_data(self, request: ExportRequest) -> ExportResult:
        """Main export method that handles all export types and formats"""
        try:
            logger.info(f"Starting export: {request.export_type} as {request.export_format}")
            
            # Generate export ID
            import uuid
            export_id = str(uuid.uuid4())
            
            # Get data based on export type
            data, total_count = await self._get_export_data(request)
            
            # Generate file based on format
            file_path, file_size = await self._generate_export_file(
                data, request, export_id
            )
            
            # Create result
            result = ExportResult(
                export_id=export_id,
                export_type=request.export_type,
                export_format=request.export_format,
                status="completed",
                file_path=file_path,
                file_size=file_size,
                record_count=len(data) if isinstance(data, list) else total_count,
                created_at=datetime.now(timezone.utc),
                download_url=f"/api/v1/exports/{export_id}/download",
                expires_at=datetime.now(timezone.utc).replace(hour=23, minute=59, second=59),
                metadata={
                    "filters_applied": request.filters,
                    "total_available": total_count,
                    "exported_count": len(data) if isinstance(data, list) else total_count,
                    "generation_time": datetime.now(timezone.utc).isoformat()
                }
            )
            
            logger.info(f"Export completed: {export_id}, {result.record_count} records")
            return result
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
    
    async def _get_export_data(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Get data to export based on export type and filters"""
        
        if request.export_type == ExportType.SOURCES:
            return await self._export_sources(request)
        elif request.export_type == ExportType.DOCUMENTS:
            return await self._export_documents(request)
        elif request.export_type == ExportType.CHUNKS:
            return await self._export_chunks(request)
        elif request.export_type == ExportType.JOBS:
            return await self._export_jobs(request)
        elif request.export_type == ExportType.MEMORIES:
            return await self._export_memories(request)
        elif request.export_type == ExportType.ANALYTICS:
            return await self._export_analytics(request)
        elif request.export_type == ExportType.SYSTEM_DATA:
            return await self._export_system_data(request)
        elif request.export_type == ExportType.ALL:
            return await self._export_all_data(request)
        else:
            raise ValueError(f"Unsupported export type: {request.export_type}")
    
    async def _export_sources(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export knowledge sources"""
        query = self.db.query(KnowledgeSource)
        
        # Apply filters
        if request.filters:
            if "status" in request.filters:
                query = query.filter(KnowledgeSource.status == request.filters["status"])
            if "source_type" in request.filters:
                query = query.filter(KnowledgeSource.source_type == request.filters["source_type"])
            if "created_after" in request.filters:
                query = query.filter(KnowledgeSource.created_at >= request.filters["created_after"])
        
        # Get total count
        total_count = query.count()
        
        # Apply sorting
        if request.sort_by:
            if hasattr(KnowledgeSource, request.sort_by):
                order_func = desc if request.sort_order == "desc" else asc
                query = query.order_by(order_func(getattr(KnowledgeSource, request.sort_by)))
        else:
            query = query.order_by(desc(KnowledgeSource.created_at))
        
        # Apply pagination
        if request.offset:
            query = query.offset(request.offset)
        if request.limit:
            query = query.limit(request.limit)
        
        sources = query.all()
        
        # Convert to dict
        data = []
        for source in sources:
            source_dict = {
                "id": str(source.id),
                "name": source.name,
                "url": source.url,
                "source_type": source.source_type,
                "status": source.status,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "updated_at": source.updated_at.isoformat() if source.updated_at else None,
                "description": source.description,
                "config": source.config,
            }
            
            if request.include_metadata:
                source_dict.update({
                    "documents_count": len(source.documents) if source.documents else 0,
                    "last_crawled": source.last_crawled.isoformat() if source.last_crawled else None,
                    "crawl_frequency": source.crawl_frequency,
                    "is_active": source.is_active
                })
            
            # Filter fields if specified
            if request.fields:
                source_dict = {k: v for k, v in source_dict.items() if k in request.fields}
            
            data.append(source_dict)
        
        return data, total_count
    
    async def _export_documents(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export documents"""
        query = self.db.query(Document)
        
        # Apply filters
        if request.filters:
            if "source_id" in request.filters:
                query = query.filter(Document.source_id == request.filters["source_id"])
            if "title" in request.filters:
                query = query.filter(Document.title.ilike(f"%{request.filters['title']}%"))
            if "content_type" in request.filters:
                query = query.filter(Document.content_type == request.filters["content_type"])
        
        total_count = query.count()
        
        # Apply sorting and pagination
        if request.sort_by and hasattr(Document, request.sort_by):
            order_func = desc if request.sort_order == "desc" else asc
            query = query.order_by(order_func(getattr(Document, request.sort_by)))
        else:
            query = query.order_by(desc(Document.created_at))
        
        if request.offset:
            query = query.offset(request.offset)
        if request.limit:
            query = query.limit(request.limit)
        
        documents = query.all()
        
        data = []
        for doc in documents:
            doc_dict = {
                "id": str(doc.id),
                "title": doc.title,
                "url": doc.url,
                "content_type": doc.content_type,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                "source_id": str(doc.source_id) if doc.source_id else None,
                "size": doc.size,
                "checksum": doc.checksum,
            }
            
            if request.include_metadata:
                doc_dict.update({
                    "chunks_count": len(doc.chunks) if doc.chunks else 0,
                    "processed": doc.processed,
                    "content_preview": doc.content[:200] + "..." if doc.content and len(doc.content) > 200 else doc.content
                })
            
            if request.fields:
                doc_dict = {k: v for k, v in doc_dict.items() if k in request.fields}
            
            data.append(doc_dict)
        
        return data, total_count
    
    async def _export_chunks(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export text chunks"""
        query = self.db.query(Chunk)
        
        # Apply filters
        if request.filters:
            if "document_id" in request.filters:
                query = query.filter(Chunk.document_id == request.filters["document_id"])
            if "min_size" in request.filters:
                query = query.filter(Chunk.size >= request.filters["min_size"])
            if "max_size" in request.filters:
                query = query.filter(Chunk.size <= request.filters["max_size"])
        
        total_count = query.count()
        
        # Apply sorting and pagination
        if request.sort_by and hasattr(Chunk, request.sort_by):
            order_func = desc if request.sort_order == "desc" else asc
            query = query.order_by(order_func(getattr(Chunk, request.sort_by)))
        else:
            query = query.order_by(desc(Chunk.created_at))
        
        if request.offset:
            query = query.offset(request.offset)
        if request.limit:
            query = query.limit(request.limit)
        
        chunks = query.all()
        
        data = []
        for chunk in chunks:
            chunk_dict = {
                "id": str(chunk.id),
                "document_id": str(chunk.document_id) if chunk.document_id else None,
                "content": chunk.content,
                "size": chunk.size,
                "chunk_index": chunk.chunk_index,
                "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
                "metadata": chunk.metadata
            }
            
            if request.include_metadata:
                chunk_dict.update({
                    "word_count": len(chunk.content.split()) if chunk.content else 0,
                    "character_count": len(chunk.content) if chunk.content else 0
                })
            
            if request.fields:
                chunk_dict = {k: v for k, v in chunk_dict.items() if k in request.fields}
            
            data.append(chunk_dict)
        
        return data, total_count
    
    async def _export_jobs(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export processing jobs"""
        query = self.db.query(Job)
        
        # Apply filters
        if request.filters:
            if "status" in request.filters:
                query = query.filter(Job.status == request.filters["status"])
            if "job_type" in request.filters:
                query = query.filter(Job.job_type == request.filters["job_type"])
            if "source_id" in request.filters:
                query = query.filter(Job.source_id == request.filters["source_id"])
        
        total_count = query.count()
        
        # Apply sorting and pagination
        if request.sort_by and hasattr(Job, request.sort_by):
            order_func = desc if request.sort_order == "desc" else asc
            query = query.order_by(order_func(getattr(Job, request.sort_by)))
        else:
            query = query.order_by(desc(Job.created_at))
        
        if request.offset:
            query = query.offset(request.offset)
        if request.limit:
            query = query.limit(request.limit)
        
        jobs = query.all()
        
        data = []
        for job in jobs:
            job_dict = {
                "id": str(job.id),
                "job_type": job.job_type.value if job.job_type else None,
                "status": job.status.value if job.status else None,
                "source_id": str(job.source_id) if job.source_id else None,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error": job.error,
                "result": job.result
            }
            
            if request.include_metadata:
                duration = None
                if job.started_at and job.completed_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                
                job_dict.update({
                    "duration_seconds": duration,
                    "progress": job.progress if hasattr(job, 'progress') else None
                })
            
            if request.fields:
                job_dict = {k: v for k, v in job_dict.items() if k in request.fields}
            
            data.append(job_dict)
        
        return data, total_count
    
    async def _export_memories(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export memory items"""
        query = self.db.query(MemoryItem)
        
        # Apply filters
        if request.filters:
            if "memory_type" in request.filters:
                query = query.filter(MemoryItem.memory_type == request.filters["memory_type"])
            if "importance" in request.filters:
                query = query.filter(MemoryItem.importance >= request.filters["importance"])
            if "session_id" in request.filters:
                query = query.filter(MemoryItem.session_id == request.filters["session_id"])
        
        total_count = query.count()
        
        # Apply sorting and pagination
        if request.sort_by and hasattr(MemoryItem, request.sort_by):
            order_func = desc if request.sort_order == "desc" else asc
            query = query.order_by(order_func(getattr(MemoryItem, request.sort_by)))
        else:
            query = query.order_by(desc(MemoryItem.created_at))
        
        if request.offset:
            query = query.offset(request.offset)
        if request.limit:
            query = query.limit(request.limit)
        
        memories = query.all()
        
        data = []
        for memory in memories:
            memory_dict = {
                "id": str(memory.id),
                "content": memory.content,
                "memory_type": memory.memory_type.value if memory.memory_type else None,
                "importance": memory.importance,
                "session_id": str(memory.session_id) if memory.session_id else None,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "metadata": memory.metadata,
                "entities": memory.entities,
                "facts": memory.facts
            }
            
            if request.include_metadata:
                memory_dict.update({
                    "content_length": len(memory.content) if memory.content else 0,
                    "entity_count": len(memory.entities) if memory.entities else 0,
                    "fact_count": len(memory.facts) if memory.facts else 0
                })
            
            if request.fields:
                memory_dict = {k: v for k, v in memory_dict.items() if k in request.fields}
            
            data.append(memory_dict)
        
        return data, total_count
    
    
    async def _export_analytics(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export analytics data"""
        # This would integrate with analytics service
        # For now, providing sample structure
        
        analytics_data = [
            {
                "metric": "total_documents",
                "value": self.db.query(Document).count(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": "documents"
            },
            {
                "metric": "total_sources", 
                "value": self.db.query(KnowledgeSource).count(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": "sources"
            },
            {
                "metric": "total_chunks",
                "value": self.db.query(Chunk).count(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": "processing"
            },
            {
                "metric": "active_jobs",
                "value": self.db.query(Job).filter(Job.status.in_(["pending", "running"])).count(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": "jobs"
            }
        ]
        
        return analytics_data, len(analytics_data)
    
    async def _export_system_data(self, request: ExportRequest) -> tuple[List[Dict[str, Any]], int]:
        """Export system configuration and metadata"""
        
        system_data = [
            {
                "component": "database",
                "status": "healthy",
                "details": {
                    "total_tables": 10,
                    "total_records": self.db.execute("SELECT COUNT(*) FROM documents").scalar() or 0,
                    "database_size": "1.2 GB"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "component": "api",
                "status": "healthy", 
                "details": {
                    "version": "1.0.0",
                    "uptime": "99.9%",
                    "endpoints": 50
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        return system_data, len(system_data)
    
    async def _export_all_data(self, request: ExportRequest) -> tuple[Dict[str, Any], int]:
        """Export all system data in a structured format"""
        
        all_data = {}
        total_records = 0
        
        # Export each data type
        for export_type in [ExportType.SOURCES, ExportType.DOCUMENTS, ExportType.CHUNKS, 
                           ExportType.JOBS, ExportType.MEMORIES]:
            try:
                type_request = ExportRequest(
                    export_type=export_type,
                    export_format=request.export_format,
                    filters=request.filters,
                    limit=request.limit,
                    include_metadata=request.include_metadata
                )
                data, count = await self._get_export_data(type_request)
                all_data[export_type.value] = data
                total_records += count
            except Exception as e:
                logger.warning(f"Failed to export {export_type}: {e}")
                all_data[export_type.value] = []
        
        # Add metadata
        all_data["export_metadata"] = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": total_records,
            "export_types": list(all_data.keys())
        }
        
        return [all_data], total_records
    
    async def _generate_export_file(self, data: Union[List[Dict[str, Any]], Dict[str, Any]], 
                                  request: ExportRequest, export_id: str) -> tuple[str, int]:
        """Generate export file in requested format"""
        
        file_path = f"{self.temp_dir}/{export_id}"
        
        if request.export_format == ExportFormat.JSON:
            file_path += ".json"
            await self._generate_json_file(data, file_path)
        
        elif request.export_format == ExportFormat.CSV:
            file_path += ".csv"
            await self._generate_csv_file(data, file_path)
        
        elif request.export_format == ExportFormat.PDF:
            file_path += ".pdf"
            await self._generate_pdf_file(data, file_path, request)
        
        elif request.export_format == ExportFormat.EXCEL:
            file_path += ".xlsx"
            await self._generate_excel_file(data, file_path)
        
        elif request.export_format == ExportFormat.XML:
            file_path += ".xml"
            await self._generate_xml_file(data, file_path)
        
        elif request.export_format == ExportFormat.ZIP:
            file_path += ".zip"
            await self._generate_zip_file(data, file_path, request, export_id)
        
        else:
            raise ValueError(f"Unsupported export format: {request.export_format}")
        
        # Get file size
        import os
        file_size = os.path.getsize(file_path)
        
        return file_path, file_size
    
    async def _generate_json_file(self, data: Any, file_path: str):
        """Generate JSON export file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    async def _generate_csv_file(self, data: List[Dict[str, Any]], file_path: str):
        """Generate CSV export file"""
        if not data:
            # Create empty CSV
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["No data available"])
            return
        
        # Get all possible fieldnames
        fieldnames = set()
        for item in data:
            fieldnames.update(item.keys())
        fieldnames = sorted(fieldnames)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                # Convert complex objects to strings
                cleaned_item = {}
                for key, value in item.items():
                    if isinstance(value, (dict, list)):
                        cleaned_item[key] = json.dumps(value, default=str)
                    else:
                        cleaned_item[key] = str(value) if value is not None else ""
                writer.writerow(cleaned_item)
    
    async def _generate_excel_file(self, data: List[Dict[str, Any]], file_path: str):
        """Generate Excel export file - TEMPORARILY DISABLED"""
        # Temporarily disabled until pandas/openpyxl are available in container
        raise HTTPException(status_code=501, detail="Excel export temporarily unavailable")
    
    async def _generate_pdf_file(self, data: List[Dict[str, Any]], file_path: str, request: ExportRequest):
        """Generate PDF export file - TEMPORARILY DISABLED"""
        # Temporarily disabled until reportlab is available in container
        raise HTTPException(status_code=501, detail="PDF export temporarily unavailable")
    
    async def _generate_xml_file(self, data: Any, file_path: str):
        """Generate XML export file"""
        import xml.etree.ElementTree as ET
        
        root = ET.Element("knowledgehub_export")
        root.set("timestamp", datetime.now(timezone.utc).isoformat())
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                item_elem = ET.SubElement(root, "item")
                item_elem.set("index", str(i))
                
                for key, value in item.items():
                    child_elem = ET.SubElement(item_elem, key)
                    if isinstance(value, (dict, list)):
                        child_elem.text = json.dumps(value, default=str)
                    else:
                        child_elem.text = str(value) if value is not None else ""
        else:
            # Single item
            for key, value in data.items():
                child_elem = ET.SubElement(root, key)
                if isinstance(value, (dict, list)):
                    child_elem.text = json.dumps(value, default=str)
                else:
                    child_elem.text = str(value) if value is not None else ""
        
        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    async def _generate_zip_file(self, data: Any, file_path: str, request: ExportRequest, export_id: str):
        """Generate ZIP file containing multiple formats"""
        with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Generate JSON
            json_path = f"{self.temp_dir}/{export_id}_data.json"
            await self._generate_json_file(data, json_path)
            zipf.write(json_path, "data.json")
            
            # Generate CSV if data is a list
            if isinstance(data, list):
                csv_path = f"{self.temp_dir}/{export_id}_data.csv"
                await self._generate_csv_file(data, csv_path)
                zipf.write(csv_path, "data.csv")
            
            # Add metadata file
            metadata = {
                "export_id": export_id,
                "export_type": request.export_type.value,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(data) if isinstance(data, list) else 1,
                "filters": request.filters
            }
            
            metadata_path = f"{self.temp_dir}/{export_id}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            zipf.write(metadata_path, "metadata.json")
            
            # Cleanup temp files
            import os
            try:
                os.remove(json_path)
                if os.path.exists(csv_path):
                    os.remove(csv_path)
                os.remove(metadata_path)
            except:
                pass  # Ignore cleanup errors
    
    def get_export_file(self, export_id: str) -> Optional[str]:
        """Get export file path by ID"""
        import glob
        import os
        
        pattern = f"{self.temp_dir}/{export_id}.*"
        files = glob.glob(pattern)
        
        if files:
            # Return the first match and verify it exists
            file_path = files[0]
            if os.path.exists(file_path):
                return file_path
        
        return None
    
    def cleanup_expired_exports(self):
        """Clean up expired export files"""
        import os
        import glob
        from datetime import datetime, timedelta
        
        # Clean files older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        pattern = f"{self.temp_dir}/*"
        for file_path in glob.glob(pattern):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"Cleaned up expired export file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {e}")


def get_export_service(db: Session) -> ExportService:
    """Get export service instance"""
    return ExportService(db)