"""Internal API endpoints for trusted services - bypasses security validation"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Dict, Any
from uuid import UUID
import logging

from ..models import get_db, Document, DocumentChunk
from ..schemas.document import DocumentResponse, DocumentCreate

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents/", response_model=DocumentResponse)
async def create_document_internal(
    document: DocumentCreate,
    db: Session = Depends(get_db)
):
    """Internal endpoint for creating documents - no security validation"""
    try:
        # Check if document already exists for this source and URL
        existing = db.query(Document).filter(
            and_(
                Document.source_id == document.source_id,
                Document.url == document.url
            )
        ).first()
        
        if existing:
            # Update existing document
            existing.title = document.title
            existing.content = document.content
            existing.content_hash = document.content_hash
            existing.status = document.status
            existing.document_metadata = document.metadata
            db.commit()
            db.refresh(existing)
            logger.info(f"Updated existing document {existing.id}")
            return DocumentResponse.from_orm(existing)
        else:
            # Create new document
            db_document = Document(
                source_id=document.source_id,
                url=document.url,
                title=document.title,
                content=document.content,
                content_hash=document.content_hash,
                status=document.status,
                document_metadata=document.metadata
            )
            db.add(db_document)
            db.commit()
            db.refresh(db_document)
            logger.info(f"Created new document {db_document.id}")
            return DocumentResponse.from_orm(db_document)
            
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chunks/batch", response_model=Dict[str, Any])
async def create_chunks_batch_internal(
    chunks: List[Dict[str, Any]],
    db: Session = Depends(get_db)
):
    """Internal endpoint for batch creating chunks - no security validation"""
    try:
        created_chunks = []
        
        for chunk_data in chunks:
            # Create chunk
            db_chunk = DocumentChunk(
                document_id=chunk_data.get("document_id"),
                source_id=chunk_data.get("source_id"),
                content=chunk_data.get("content"),
                chunk_type=chunk_data.get("chunk_type", "text"),
                chunk_index=chunk_data.get("chunk_index", 0),
                parent_heading=chunk_data.get("parent_heading"),
                metadata=chunk_data.get("metadata", {})
            )
            db.add(db_chunk)
            db.flush()  # Get the ID without committing
            
            created_chunks.append({
                "id": str(db_chunk.id),
                "document_id": str(db_chunk.document_id) if db_chunk.document_id else None,
                "source_id": str(db_chunk.source_id) if db_chunk.source_id else None
            })
        
        db.commit()
        logger.info(f"Created {len(created_chunks)} chunks via internal API")
        
        return {
            "status": "success",
            "chunks": created_chunks,
            "count": len(created_chunks)
        }
        
    except Exception as e:
        logger.error(f"Error creating chunks batch: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))