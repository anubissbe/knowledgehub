"""Document schemas"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class DocumentBase(BaseModel):
    """Base document schema"""
    source_id: UUID
    url: str
    title: str
    content: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document"""
    status: Optional[str] = "indexed"


class DocumentUpdate(BaseModel):
    """Schema for updating a document"""
    title: Optional[str] = None
    content: Optional[str] = None
    content_hash: Optional[str] = None
    status: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    """Schema for document response"""
    id: UUID
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True