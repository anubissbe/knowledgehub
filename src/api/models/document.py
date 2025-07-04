"""Document and chunk models"""

from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum
from typing import Optional, Dict, Any, List

from .base import Base


class ChunkType(str, enum.Enum):
    """Type of document chunk"""
    TEXT = "TEXT"
    CODE = "CODE"
    TABLE = "TABLE"
    LIST = "LIST"
    HEADING = "HEADING"


class Document(Base):
    """Model for scraped documents"""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_sources.id", ondelete="CASCADE"), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(Text)
    content = Column(Text)
    content_hash = Column(String(64))
    document_metadata = Column("metadata", JSON, default={})
    status = Column(String(50), default="pending")
    indexed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source = relationship("KnowledgeSource", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, url='{self.url}', title='{self.title}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "source_id": str(self.source_id),
            "url": self.url,
            "title": self.title,
            "content_hash": self.content_hash,
            "metadata": self.document_metadata,
            "status": self.status,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentChunk(Base):
    """Model for document chunks (for vector storage tracking)"""
    
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_type: ChunkType = Column(
        SQLEnum(ChunkType, name="chunk_type", values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    content = Column(Text, nullable=False)
    embedding_id = Column(String(255))  # ID in vector database
    parent_heading = Column(Text)
    chunk_metadata = Column("metadata", JSON, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, type={self.chunk_type})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "document_id": str(self.document_id),
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type.value if isinstance(self.chunk_type, ChunkType) else self.chunk_type,
            "content": self.content,
            "embedding_id": self.embedding_id,
            "parent_heading": self.parent_heading,
            "metadata": self.chunk_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# Alias for backward compatibility
Chunk = DocumentChunk