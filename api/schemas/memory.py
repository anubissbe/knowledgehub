"""Memory schemas"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID

from ..security import InputSanitizer


class MemoryCreate(BaseModel):
    """Schema for creating a memory"""
    content: str = Field(..., min_length=1)
    content_hash: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('content')
    def sanitize_content(cls, v):
        """Sanitize memory content"""
        return InputSanitizer.sanitize_text(v, max_length=10000, allow_html=False)
    
    @field_validator('content_hash')
    def sanitize_content_hash(cls, v):
        """Sanitize content hash"""
        if v is not None:
            # Hash should be alphanumeric only
            sanitized = InputSanitizer.sanitize_text(v, max_length=64, allow_html=False)
            if not sanitized.replace('_', '').replace('-', '').isalnum():
                raise ValueError("Content hash must be alphanumeric")
            return sanitized
        return v
    
    @field_validator('tags')
    def sanitize_tags(cls, v):
        """Sanitize memory tags"""
        if v is not None:
            return InputSanitizer.sanitize_list(v)
        return v
    
    @field_validator('metadata')
    def sanitize_metadata(cls, v):
        """Sanitize memory metadata"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v


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
    
    model_config = ConfigDict(from_attributes=True,
                            populate_by_name=True)