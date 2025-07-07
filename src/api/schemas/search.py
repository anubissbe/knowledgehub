"""Search schemas"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

from ..security import InputSanitizer


class SearchType(str, Enum):
    """Search type enum"""
    HYBRID = "hybrid"
    VECTOR = "vector"
    KEYWORD = "keyword"


class SearchQuery(BaseModel):
    """Schema for search query"""
    query: str = Field(..., min_length=1)
    source_filter: Optional[str] = None
    limit: int = Field(10, ge=1, le=100)
    include_metadata: bool = True
    search_type: SearchType = SearchType.HYBRID
    filters: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def sanitize_query(cls, v):
        """Sanitize search query to prevent injection attacks"""
        # Allow longer queries for search but still sanitize
        return InputSanitizer.sanitize_text(v, max_length=2000, allow_html=False)
    
    @validator('source_filter')
    def sanitize_source_filter(cls, v):
        """Sanitize source filter"""
        if v is not None:
            return InputSanitizer.sanitize_text(v, max_length=255, allow_html=False)
        return v
    
    @validator('filters')
    def sanitize_filters(cls, v):
        """Sanitize search filters"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v


class SearchResult(BaseModel):
    """Schema for individual search result"""
    content: str
    source_name: str
    url: str
    score: float
    chunk_type: str
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('content')
    def sanitize_content(cls, v):
        """Sanitize search result content"""
        return InputSanitizer.sanitize_text(v, max_length=5000, allow_html=False)
    
    @validator('source_name')
    def sanitize_source_name(cls, v):
        """Sanitize source name in results"""
        return InputSanitizer.sanitize_text(v, max_length=255, allow_html=False)
    
    @validator('metadata')
    def sanitize_metadata(cls, v):
        """Sanitize metadata in results"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v


class SearchResponse(BaseModel):
    """Schema for search response"""
    query: str
    results: List[SearchResult]
    total: int
    search_time_ms: float
    
    @validator('query')
    def sanitize_query_output(cls, v):
        """Sanitize query in response output"""
        return InputSanitizer.sanitize_text(v, max_length=2000, allow_html=False)