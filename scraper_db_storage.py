#!/usr/bin/env python3
"""
Direct database storage for scraper - bypasses API authentication
"""

import os
import sys
sys.path.append('/opt/projects/knowledgehub')

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from api.models import Document, DocumentChunk, KnowledgeSource
from uuid import uuid4
import hashlib
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "postgresql://knowledgehub:knowledgehub123@localhost:5433/knowledgehub"

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def store_scraped_document(source_id: str, url: str, title: str, content: str, metadata: dict = None):
    """Store a scraped document directly to the database"""
    db = SessionLocal()
    try:
        # Check if document already exists
        existing = db.query(Document).filter(
            Document.source_id == source_id,
            Document.url == url
        ).first()
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        if existing:
            # Update existing document
            existing.title = title
            existing.content = content
            existing.content_hash = content_hash
            existing.status = 'indexed'
            existing.document_metadata = metadata or {}
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            logger.info(f"Updated existing document {existing.id}")
            return str(existing.id)
        else:
            # Create new document
            doc = Document(
                id=uuid4(),
                source_id=source_id,
                url=url,
                title=title,
                content=content,
                content_hash=content_hash,
                status='indexed',
                document_metadata=metadata or {},
                created_at=datetime.utcnow()
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
            
            # Update source stats
            source = db.query(KnowledgeSource).filter(KnowledgeSource.id == source_id).first()
            if source:
                if not source.stats:
                    source.stats = {"documents": 0, "chunks": 0, "errors": 0}
                source.stats["documents"] = source.stats.get("documents", 0) + 1
                db.commit()
                logger.info(f"Updated source stats: {source.stats['documents']} documents")
            
            logger.info(f"Created new document {doc.id}")
            return str(doc.id)
            
    except Exception as e:
        logger.error(f"Error storing document: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    # Test the function
    doc_id = store_scraped_document(
        source_id=str(uuid4()),  # Generate valid UUID
        url="https://example.com/test",
        title="Test Document",
        content="This is test content",
        metadata={"scraped_at": datetime.utcnow().isoformat()}
    )
    print(f"Stored document with ID: {doc_id}")