"""Memory schemas"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class MemoryCreate(BaseModel):
    """Schema for creating a memory"""
    content: str = Field(..., min_length=1)
    content_hash: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Schema for memory response"""
    id: UUID
    content: str
    content_hash: str
    tags: List[str]
    metadata: Dict[str, Any] = Field(alias="meta_data")
    embedding_id: Optional[str]
    access_count: int
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    
    class Config:
        from_attributes = True
        populate_by_name = True