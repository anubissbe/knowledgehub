"""Documents router for managing indexed documents"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from ..models import get_db, Document, DocumentChunk
from ..schemas.document import DocumentResponse, DocumentCreate, DocumentUpdate
from ..dependencies import get_current_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    source_id: Optional[UUID] = Query(None, description="Filter by source ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=10000),
    db: Session = Depends(get_db)
):
    """List documents with optional filtering"""
    try:
        query = db.query(Document)
        
        # Apply filters
        if source_id:
            query = query.filter(Document.source_id == source_id)
        if status:
            query = query.filter(Document.status == status)
            
        # Get total count
        total = query.count()
        
        # Get paginated results
        documents = query.offset(skip).limit(limit).all()
        
        # Convert to response format
        return {
            "documents": [
                {
                    "id": str(doc.id),
                    "source_id": str(doc.source_id),
                    "url": doc.url,
                    "title": doc.title,
                    "content_hash": doc.content_hash,
                    "status": doc.status,
                    "created_at": doc.created_at.isoformat(),
                    "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
                    "metadata": doc.document_metadata or {}
                }
                for doc in documents
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific document by ID"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    return DocumentResponse.from_orm(document)


@router.post("/", response_model=DocumentResponse)
async def create_document(
    document: DocumentCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new document"""
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
            existing.metadata = document.metadata
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return DocumentResponse.from_orm(existing)
            
        # Create new document
        db_document = Document(
            source_id=document.source_id,
            url=document.url,
            title=document.title,
            content=document.content,
            content_hash=document.content_hash,
            metadata=document.metadata,
            created_at=datetime.utcnow()
        )
        
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        return DocumentResponse.from_orm(db_document)
        
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: UUID,
    document_update: DocumentUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update an existing document"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Update fields
    if document_update.title is not None:
        document.title = document_update.title
    if document_update.content is not None:
        document.content = document_update.content
    if document_update.content_hash is not None:
        document.content_hash = document_update.content_hash
    if document_update.status is not None:
        document.status = document_update.status
    if document_update.metadata is not None:
        document.metadata = document_update.metadata
        
    document.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(document)
    
    return DocumentResponse.from_orm(document)


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a document and its chunks"""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Delete associated chunks first
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    
    # Delete document
    db.delete(document)
    db.commit()
    
    return {"message": "Document deleted successfully"}


@router.get("/{document_id}/chunks", response_model=Dict[str, Any])
async def get_document_chunks(
    document_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get chunks for a specific document"""
    # Verify document exists
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
        
    # Get chunks
    query = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id)
    total = query.count()
    chunks = query.offset(skip).limit(limit).all()
    
    return {
        "chunks": [
            {
                "id": str(chunk.id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata or {},
                "created_at": chunk.created_at.isoformat()
            }
            for chunk in chunks
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }