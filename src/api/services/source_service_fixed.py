"""Source management service - Fixed version"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from uuid import UUID
import json

from ..models import get_db
from ..models.source import KnowledgeSource as Source
from ..schemas.source import SourceCreate, SourceUpdate


class SourceService:
    """Service for managing knowledge sources"""
    
    def __init__(self):
        """Initialize service"""
        self._db = None
    
    @property
    def db(self) -> Session:
        """Get database session"""
        if self._db is None:
            # Get a new session
            self._db = next(get_db())
        return self._db
    
    def __del__(self):
        """Clean up database session"""
        if self._db:
            self._db.close()
    
    async def list_sources(self, skip: int = 0, limit: int = 100, status: Optional[str] = None) -> List[Source]:
        """List all knowledge sources"""
        query = self.db.query(Source)
        if status:
            query = query.filter(Source.status == status)
        return query.offset(skip).limit(limit).all()
    
    async def count_sources(self, status: Optional[str] = None) -> int:
        """Count knowledge sources"""
        query = self.db.query(Source)
        if status:
            query = query.filter(Source.status == status)
        return query.count()
    
    async def get_by_url(self, url: str) -> Optional[Source]:
        """Get source by URL"""
        return self.db.query(Source).filter(Source.url == url).first()
    
    async def get_by_id(self, source_id: UUID) -> Optional[Source]:
        """Get source by ID"""
        return self.db.query(Source).filter(Source.id == source_id).first()
    
    async def create(self, source: SourceCreate) -> Source:
        """Create a new knowledge source"""
        # Create source record
        db_source = Source(
            url=source.url,
            name=source.name,
            source_type=source.type,
            authentication=source.authentication or {},
            crawl_config=source.crawl_config or {
                "max_depth": 2,
                "max_pages": 100,
                "follow_patterns": [],
                "exclude_patterns": []
            },
            refresh_interval=source.refresh_interval,
            status="pending"
        )
        
        try:
            self.db.add(db_source)
            self.db.commit()
            self.db.refresh(db_source)
            
            # Queue initial crawl job (if message queue available)
            from .message_queue import message_queue
            if message_queue and message_queue.client:
                await message_queue.publish(
                    "crawl_jobs",
                    json.dumps({
                        "job_type": "initial_crawl",
                        "source_id": str(db_source.id),
                        "url": db_source.url,
                        "config": db_source.crawl_config
                    })
                )
            
            return db_source
            
        except IntegrityError:
            self.db.rollback()
            raise ValueError("Source with this URL already exists")
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def update(self, source_id: UUID, update_data: SourceUpdate) -> Optional[Source]:
        """Update a knowledge source"""
        source = await self.get_by_id(source_id)
        if not source:
            return None
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(source, field, value)
        
        self.db.commit()
        self.db.refresh(source)
        return source
    
    async def delete(self, source_id: UUID) -> bool:
        """Delete a knowledge source"""
        source = await self.get_by_id(source_id)
        if not source:
            return False
        
        self.db.delete(source)
        self.db.commit()
        return True