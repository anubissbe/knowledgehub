"""Search history model for analytics"""

from sqlalchemy import Column, String, DateTime, JSON, Integer, Float, Text
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

from .base import Base


class SearchHistory(Base):
    """Model for tracking search queries and performance"""
    
    __tablename__ = "search_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    results_count = Column(Integer, default=0)
    search_type = Column(String(50), default="hybrid")
    filters = Column(JSON, default={})
    execution_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SearchHistory(id={self.id}, query='{self.query[:50]}...', results={self.results_count})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": str(self.id),
            "query": self.query,
            "results_count": self.results_count,
            "search_type": self.search_type,
            "filters": self.filters,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }