"""Document chunks router"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import func, and_
from sqlalchemy.exc import IntegrityError
from uuid import UUID
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from ..dependencies import get_db
from ..models.document import DocumentChunk, ChunkType, Document
from ..models.source import KnowledgeSource

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", status_code=201)
async def create_chunk(
    chunk_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create a new document chunk"""
    try:
        # Extract data
        document_id = chunk_data.get("document_id")
        source_id = chunk_data.get("source_id")  # Optional, for backward compatibility
        content = chunk_data.get("content")
        chunk_type = chunk_data.get("chunk_type", "text")
        chunk_index = chunk_data.get("chunk_index", 0)
        metadata = chunk_data.get("metadata", {})
        embedding_id = chunk_data.get("embedding_id")
        parent_heading = chunk_data.get("parent_heading")
        
        # If source_id is provided but not document_id, create or find document
        if source_id and not document_id:
            url = metadata.get("url", f"unknown_{datetime.utcnow().isoformat()}")
            
            # Find or create document with better error handling
            document = db.query(Document).filter(
                and_(Document.source_id == source_id, Document.url == url)
            ).first()
            
            # Update content hash if document exists and hash changed
            if document and chunk_data.get("content_hash"):
                if document.content_hash != chunk_data.get("content_hash"):
                    document.content_hash = chunk_data.get("content_hash")
                    document.updated_at = datetime.utcnow()
                    db.commit()
            
            if not document:
                try:
                    document = Document(
                        source_id=source_id,
                        url=url,
                        title=metadata.get("title", ""),
                        content="",  # Will be populated from chunks
                        content_hash=chunk_data.get("content_hash"),
                        metadata=metadata,
                        created_at=datetime.utcnow()
                    )
                    db.add(document)
                    db.flush()  # Get the ID without committing
                except Exception as e:
                    # If document creation fails (e.g., unique constraint), try to find it again
                    db.rollback()
                    document = db.query(Document).filter(
                        and_(Document.source_id == source_id, Document.url == url)
                    ).first()
                    if not document:
                        raise HTTPException(status_code=500, detail=f"Failed to create or find document: {str(e)}")
            
            document_id = document.id
        
        # Validate document exists
        if not document_id:
            raise HTTPException(status_code=400, detail="Either document_id or source_id must be provided")
            
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Convert chunk_type string to enum
        try:
            # Map string to proper enum member
            chunk_type_lower = chunk_type.lower()
            if chunk_type_lower == 'text':
                chunk_type_enum = ChunkType.TEXT
            elif chunk_type_lower == 'code':
                chunk_type_enum = ChunkType.CODE
            elif chunk_type_lower == 'table':
                chunk_type_enum = ChunkType.TABLE
            elif chunk_type_lower == 'list':
                chunk_type_enum = ChunkType.LIST
            elif chunk_type_lower == 'heading':
                chunk_type_enum = ChunkType.HEADING
            else:
                chunk_type_enum = ChunkType.TEXT
        except Exception:
            chunk_type_enum = ChunkType.TEXT
        
        # Check if chunk already exists
        existing_chunk = db.query(DocumentChunk).filter(
            and_(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_index == chunk_index
            )
        ).first()
        
        if existing_chunk:
            # Update existing chunk
            existing_chunk.content = content
            existing_chunk.chunk_type = chunk_type_enum
            existing_chunk.chunk_metadata = metadata
            existing_chunk.embedding_id = embedding_id
            existing_chunk.parent_heading = parent_heading
            existing_chunk.updated_at = datetime.utcnow()
            chunk = existing_chunk
            logger.info(f"Updated existing chunk {chunk.id} for document {document_id}")
        else:
            # Create new chunk
            chunk = DocumentChunk(
                document_id=document_id,
                content=content,
                chunk_type=chunk_type_enum,
                chunk_index=chunk_index,
                chunk_metadata=metadata,
                embedding_id=embedding_id,
                parent_heading=parent_heading,
                created_at=datetime.utcnow()
            )
            db.add(chunk)
        
        try:
            db.commit()
            db.refresh(chunk)
            
            # Update source stats
            update_source_stats(db, document.source_id)
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating chunk: {e}")
            raise HTTPException(status_code=409, detail=f"Chunk creation failed due to constraint violation: {str(e)}")
        except Exception as e:
            db.rollback()
            logger.error(f"Unexpected error creating chunk: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create chunk: {str(e)}")
        
        return {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "source_id": str(document.source_id),
            "chunk_type": chunk.chunk_type.value,
            "chunk_index": chunk.chunk_index,
            "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
            "created_at": chunk.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chunk: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", status_code=201)
async def create_chunks_batch(
    chunks: List[Dict[str, Any]],
    db: Session = Depends(get_db)
):
    """Create multiple chunks in a batch"""
    created_chunks = []
    source_ids = set()
    documents_cache = {}
    
    try:
        for chunk_data in chunks:
            # Extract data
            document_id = chunk_data.get("document_id")
            source_id = chunk_data.get("source_id")
            content = chunk_data.get("content")
            chunk_type = chunk_data.get("chunk_type", "text")
            chunk_index = chunk_data.get("chunk_index", 0)
            metadata = chunk_data.get("metadata", {})
            embedding_id = chunk_data.get("embedding_id")
            parent_heading = chunk_data.get("parent_heading")
            
            # If source_id is provided but not document_id, create or find document
            if source_id and not document_id:
                url = metadata.get("url", f"unknown_{datetime.utcnow().isoformat()}")
                cache_key = f"{source_id}:{url}"
                
                if cache_key in documents_cache:
                    document = documents_cache[cache_key]
                else:
                    # Find or create document
                    document = db.query(Document).filter(
                        and_(Document.source_id == source_id, Document.url == url)
                    ).first()
                    
                    if not document:
                        document = Document(
                            source_id=source_id,
                            url=url,
                            title=metadata.get("title", ""),
                            content="",
                            metadata=metadata,
                            created_at=datetime.utcnow()
                        )
                        db.add(document)
                        db.flush()
                    
                    documents_cache[cache_key] = document
                
                document_id = document.id
                source_ids.add(source_id)
            elif document_id:
                # Get source_id from document
                if document_id not in documents_cache:
                    document = db.query(Document).filter(Document.id == document_id).first()
                    if document:
                        documents_cache[document_id] = document
                        source_ids.add(document.source_id)
            
            # Convert chunk_type string to enum
            try:
                # Map string to proper enum member
                chunk_type_lower = chunk_type.lower()
                if chunk_type_lower == 'text':
                    chunk_type_enum = ChunkType.TEXT
                elif chunk_type_lower == 'code':
                    chunk_type_enum = ChunkType.CODE
                elif chunk_type_lower == 'table':
                    chunk_type_enum = ChunkType.TABLE
                elif chunk_type_lower == 'list':
                    chunk_type_enum = ChunkType.LIST
                elif chunk_type_lower == 'heading':
                    chunk_type_enum = ChunkType.HEADING
                else:
                    chunk_type_enum = ChunkType.TEXT
            except Exception:
                chunk_type_enum = ChunkType.TEXT
            
            # Check if chunk already exists
            existing_chunk = db.query(DocumentChunk).filter(
                and_(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.chunk_index == chunk_index
                )
            ).first()
            
            if existing_chunk:
                # Update existing chunk
                existing_chunk.content = content
                existing_chunk.chunk_type = chunk_type_enum
                existing_chunk.chunk_metadata = metadata
                existing_chunk.embedding_id = embedding_id
                existing_chunk.parent_heading = parent_heading
                existing_chunk.updated_at = datetime.utcnow()
                chunk = existing_chunk
            else:
                # Create new chunk
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=content,
                    chunk_type=chunk_type_enum,
                    chunk_index=chunk_index,
                    chunk_metadata=metadata,
                    embedding_id=embedding_id,
                    parent_heading=parent_heading,
                    created_at=datetime.utcnow()
                )
                db.add(chunk)
            
            created_chunks.append(chunk)
        
        # Commit all chunks
        db.commit()
        
        # Update stats for all affected sources
        for source_id in source_ids:
            update_source_stats(db, source_id)
        
        return {
            "created": len(created_chunks),
            "chunks": [
                {
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in created_chunks
            ]
        }
        
    except Exception as e:
        logger.error(f"Error creating chunks batch: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating chunks: {str(e)}")


@router.get("/")
async def list_chunks(
    source_id: Optional[UUID] = Query(None),
    document_id: Optional[UUID] = Query(None),
    chunk_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List chunks with optional filtering"""
    try:
        query = db.query(DocumentChunk).join(Document)
        
        if source_id:
            query = query.filter(Document.source_id == source_id)
        
        if document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        
        if chunk_type:
            try:
                chunk_type_enum = ChunkType[chunk_type.upper()]
                query = query.filter(DocumentChunk.chunk_type == chunk_type_enum)
            except KeyError:
                pass
        
        total = query.count()
        chunks = query.order_by(DocumentChunk.created_at.desc()).offset(skip).limit(limit).all()
        
        # Get document info for chunks
        doc_ids = list(set(chunk.document_id for chunk in chunks))
        documents = {
            doc.id: doc 
            for doc in db.query(Document).filter(Document.id.in_(doc_ids)).all()
        } if doc_ids else {}
        
        return {
            "chunks": [
                {
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "source_id": str(documents[chunk.document_id].source_id) if chunk.document_id in documents else None,
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_index": chunk.chunk_index,
                    "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                    "embedding_id": chunk.embedding_id,
                    "parent_heading": chunk.parent_heading,
                    "created_at": chunk.created_at.isoformat()
                }
                for chunk in chunks
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error listing chunks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats")
async def get_chunk_stats(
    source_id: Optional[UUID] = Query(None),
    document_id: Optional[UUID] = Query(None),
    db: Session = Depends(get_db)
):
    """Get chunk statistics"""
    try:
        query = db.query(
            DocumentChunk.chunk_type,
            func.count(DocumentChunk.id).label('count')
        )
        
        if source_id or document_id:
            query = query.join(Document)
            
        if source_id:
            query = query.filter(Document.source_id == source_id)
            
        if document_id:
            query = query.filter(DocumentChunk.document_id == document_id)
        
        stats = query.group_by(DocumentChunk.chunk_type).all()
        
        return {
            "stats": {
                chunk_type.value: count 
                for chunk_type, count in stats
            },
            "total": sum(count for _, count in stats)
        }
        
    except Exception as e:
        logger.error(f"Error getting chunk stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{chunk_id}")
async def update_chunk(
    chunk_id: UUID,
    update_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update a chunk (mainly for updating embedding_id)"""
    try:
        chunk = db.query(DocumentChunk).filter(DocumentChunk.id == chunk_id).first()
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Update allowed fields
        if "embedding_id" in update_data:
            chunk.embedding_id = update_data["embedding_id"]
        
        chunk.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Chunk updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chunk: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/batch/embedding-ids")
async def update_chunk_embedding_ids_batch(
    updates: List[Dict[str, str]],  # List of {"chunk_id": "...", "embedding_id": "..."}
    db: Session = Depends(get_db)
):
    """Update multiple chunks with embedding IDs in a batch"""
    try:
        updated_count = 0
        
        for update in updates:
            chunk_id = update.get("chunk_id")
            embedding_id = update.get("embedding_id")
            
            if not chunk_id or not embedding_id:
                continue
                
            chunk = db.query(DocumentChunk).filter(
                DocumentChunk.id == UUID(chunk_id)
            ).first()
            
            if chunk:
                chunk.embedding_id = embedding_id
                chunk.updated_at = datetime.utcnow()
                updated_count += 1
        
        db.commit()
        
        return {
            "message": f"Updated {updated_count} chunks successfully",
            "updated": updated_count
        }
        
    except Exception as e:
        logger.error(f"Error updating chunks batch: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating chunks: {str(e)}")


def update_source_stats(db: Session, source_id: UUID):
    """Update source statistics based on chunks"""
    try:
        # Get chunk count for this source
        chunk_count = db.query(func.count(DocumentChunk.id)).join(Document).filter(
            Document.source_id == source_id
        ).scalar()
        
        # Get document count for this source
        doc_count = db.query(func.count(Document.id)).filter(
            Document.source_id == source_id
        ).scalar()
        
        # Get unique URLs count (as a measure of unique documents)
        unique_urls = db.query(func.count(func.distinct(Document.url))).filter(
            Document.source_id == source_id
        ).scalar()
        
        # Update source
        source = db.query(KnowledgeSource).filter(KnowledgeSource.id == source_id).first()
        if source:
            if not source.stats:
                source.stats = {}
            
            source.stats["chunks"] = chunk_count or 0
            source.stats["documents"] = doc_count or 0
            source.stats["unique_urls"] = unique_urls or 0
            source.stats["errors"] = source.stats.get("errors", 0)
            source.updated_at = datetime.utcnow()
            
            # Flag the stats column as modified for SQLAlchemy to track the change
            flag_modified(source, "stats")
            
            db.commit()
            logger.info(f"Updated stats for source {source_id}: chunks={chunk_count}, docs={doc_count}, unique_urls={unique_urls}")
            
    except Exception as e:
        logger.error(f"Error updating source stats: {e}")
        db.rollback()
