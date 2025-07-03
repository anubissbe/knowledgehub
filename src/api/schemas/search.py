"""Search schemas"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


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


class SearchResult(BaseModel):
    """Schema for individual search result"""
    content: str
    source_name: str
    url: str
    score: float
    chunk_type: str
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Schema for search response"""
    query: str
    results: List[SearchResult]
    total: int
    search_time_ms: float