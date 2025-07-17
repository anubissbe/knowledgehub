"""Knowledge Source Model for Documentation Sources.

This module defines the core data model for knowledge sources, which represent
external documentation sources like websites, GitHub repositories, or file uploads
that can be crawled and indexed by the KnowledgeHub system.

The KnowledgeSource model tracks:
    - Source metadata (name, URL, configuration)
    - Processing status and lifecycle management
    - Crawling statistics and error tracking
    - Relationships to documents and jobs
    - Scheduling and automation settings

Supported Source Types:
    - Documentation websites (docs.python.org, fastapi docs, etc.)
    - GitHub repositories (README files, wiki pages)
    - File uploads (PDF, text files, documentation)
    - API documentation (OpenAPI, GraphQL schemas)

Lifecycle States:
    PENDING → CRAWLING → INDEXING → COMPLETED
                ↓           ↓
              ERROR      PAUSED

Example:
    source = KnowledgeSource(
        name="FastAPI Documentation",
        url="https://fastapi.tiangolo.com/",
        config={
            "max_depth": 3,
            "follow_patterns": [r"https://fastapi\.tiangolo\.com/.*"],
            "exclude_patterns": [r"\.(pdf|jpg|png)$"]
        }
    )
"""

from sqlalchemy import Column, String, DateTime, JSON, Text, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum
from typing import Optional, Dict, Any, List

from .base import Base


class SourceStatus(str, enum.Enum):
    """Enumeration of possible knowledge source processing states.
    
    This enum tracks the current state of a knowledge source throughout
    its processing lifecycle. The status determines what operations can
    be performed and provides visibility into the system's progress.
    
    States:
        PENDING: Source created but not yet processed
        CRAWLING: Actively scraping content from the source
        INDEXING: Processing scraped content into searchable chunks
        COMPLETED: Successfully processed and ready for search
        ERROR: Processing failed - check error logs
        PAUSED: Processing temporarily stopped (manual or automatic)
        
    State Transitions:
        - PENDING → CRAWLING (when job starts)
        - CRAWLING → INDEXING (when crawl completes)
        - INDEXING → COMPLETED (when processing finishes)
        - Any state → ERROR (on failure)
        - Any state → PAUSED (manual intervention)
        - PAUSED → CRAWLING (when resumed)
        
    Note:
        The ERROR state requires manual intervention to retry processing.
        The PAUSED state can be resumed automatically or manually.
    """
    PENDING = "pending"
    CRAWLING = "crawling"
    INDEXING = "indexing"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class KnowledgeSource(Base):
    """SQLAlchemy model representing a knowledge source.
    
    A knowledge source is any external source of documentation or content
    that can be crawled, processed, and indexed for search. This model
    stores all metadata, configuration, and status tracking information.
    
    The model includes:
    - Basic metadata (name, URL, timestamps)
    - Processing status and lifecycle tracking
    - Flexible JSON configuration for crawling parameters
    - Statistics tracking (document counts, error rates)
    - Relationships to documents and processing jobs
    
    Configuration Options:
        - max_depth: Maximum crawl depth (default: 2)
        - max_pages: Maximum pages to crawl (default: 1000)
        - follow_patterns: Regex patterns for URLs to include
        - exclude_patterns: Regex patterns for URLs to exclude
        - crawl_delay: Delay between requests in seconds
        - custom_headers: Additional HTTP headers for crawling
        - authentication: Auth credentials if required
        
    Statistics Tracked:
        - documents: Number of documents created
        - chunks: Number of content chunks generated
        - errors: Number of processing errors encountered
        - last_crawl_duration: Time taken for last crawl
        - success_rate: Percentage of successful page crawls
        
    Database Relationships:
        - documents: One-to-many with Document model
        - jobs: One-to-many with ScrapingJob model
        
    Example Configuration:
        {
            "max_depth": 3,
            "max_pages": 500,
            "follow_patterns": [r"https://docs\.example\.com/.*"],
            "exclude_patterns": [r"\.(pdf|jpg|png)$", r"/api/"],
            "crawl_delay": 1.0,
            "custom_headers": {"User-Agent": "KnowledgeHub/1.0"}
        }
    """
    
    __tablename__ = "knowledge_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    url = Column(Text, nullable=False, unique=True)
    type = Column(String(50), nullable=False, default="website")
    status: SourceStatus = Column(
        SQLEnum(SourceStatus, name="source_status", values_callable=lambda obj: [e.value for e in obj]),
        default=SourceStatus.PENDING,
        nullable=False
    )
    config = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_scraped_at = Column(DateTime(timezone=True), nullable=True)
    stats = Column(JSON, default={"documents": 0, "chunks": 0, "errors": 0})
    
    # SQLAlchemy Relationships
    # These define foreign key relationships with cascade delete for cleanup
    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")
    jobs = relationship("ScrapingJob", back_populates="source", cascade="all, delete-orphan")
    
    @property
    def document_count(self) -> int:
        """Return the count of documents associated with this source."""
        return self.stats.get("documents", 0) if self.stats else 0

    def __repr__(self) -> str:
        """Return string representation of the knowledge source.
        
        Provides a concise, human-readable representation of the instance
        for debugging and logging purposes.
        
        Returns:
            str: String representation including key identifiers and status
        """
        return f"<KnowledgeSource(id={self.id}, name='{self.name}', url='{self.url}', status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the knowledge source to a dictionary for API responses.
        
        This method serializes the model instance into a JSON-compatible
        dictionary format suitable for REST API responses. It handles
        type conversions, date formatting, and enum serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation containing:
                - id (str): UUID as string
                - name (str): Human-readable source name
                - url (str): Source URL
                - status (str): Current processing status
                - config (dict): Crawling configuration
                - created_at (str): ISO formatted creation timestamp
                - updated_at (str): ISO formatted last update timestamp
                - last_scraped_at (str|None): ISO formatted last crawl timestamp
                - stats (dict): Processing statistics
                
        Note:
            Timestamps are converted to ISO format strings for JSON compatibility.
            UUIDs are converted to strings. Enum values use their string representation.
            
        Example Output:
            {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "FastAPI Docs",
                "url": "https://fastapi.tiangolo.com/",
                "status": "completed",
                "config": {"max_depth": 2},
                "created_at": "2025-01-01T12:00:00Z",
                "updated_at": "2025-01-02T12:00:00Z",
                "last_scraped_at": "2025-01-02T11:30:00Z",
                "stats": {"documents": 150, "chunks": 1200, "errors": 0}
            }
        """
        return {
            "id": str(self.id),
            "name": self.name,
            "url": self.url,
            "status": self.status.value if isinstance(self.status, SourceStatus) else self.status,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_scraped_at": self.last_scraped_at.isoformat() if self.last_scraped_at else None,
            "stats": self.stats,
        }