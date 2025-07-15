"""Source management service - Fixed version"""

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from uuid import UUID
import json
import logging

from ..models import get_db
from ..models.source import KnowledgeSource as Source
from ..schemas.source import SourceCreate, SourceUpdate

logger = logging.getLogger(__name__)


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
        # Convert URL object to string if needed
        url_str = str(url)
        return self.db.query(Source).filter(Source.url == url_str).first()
    
    async def get_by_id(self, source_id: UUID) -> Optional[Source]:
        """Get source by ID"""
        return self.db.query(Source).filter(Source.id == source_id).first()
    
    async def create(self, source: SourceCreate) -> Source:
        """Create a new knowledge source"""
        # Create source record
        from ..models.knowledge_source import SourceStatus
        config = source.config or {}
        # Add default values if not provided
        if "max_depth" not in config:
            config["max_depth"] = 2
        if "max_pages" not in config:
            config["max_pages"] = 100
        if "follow_patterns" not in config:
            config["follow_patterns"] = []
        if "exclude_patterns" not in config:
            config["exclude_patterns"] = []
            
        db_source = Source(
            url=str(source.url),
            name=source.name,
            config=config,
            status=SourceStatus.PENDING
        )
        
        try:
            self.db.add(db_source)
            self.db.commit()
            self.db.refresh(db_source)
            
            # Queue initial crawl job (if message queue available)
            try:
                from .message_queue import message_queue
                if message_queue and hasattr(message_queue, 'client') and message_queue.client:
                    await message_queue.publish(
                        "crawl_jobs",
                        json.dumps({
                            "job_type": "initial_crawl",
                            "source_id": str(db_source.id),
                            "url": db_source.url,
                            "config": db_source.config
                        })
                    )
                else:
                    logger.warning("Message queue not available, skipping job queue")
            except Exception as mq_error:
                logger.error(f"Failed to queue crawl job: {mq_error}")
                # Don't fail the entire operation if message queue fails
            
            return db_source
            
        except IntegrityError:
            self.db.rollback()
            raise ValueError("Source with this URL already exists")
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def create_from_dict(self, source_data: dict) -> Source:
        """Create a new knowledge source from dictionary"""
        from ..models.knowledge_source import SourceStatus
        config = source_data.get("config", {})
        # Add default values if not provided
        if "max_depth" not in config:
            config["max_depth"] = 2
        if "max_pages" not in config:
            config["max_pages"] = 100
        if "follow_patterns" not in config:
            config["follow_patterns"] = []
        if "exclude_patterns" not in config:
            config["exclude_patterns"] = []
            
        db_source = Source(
            url=source_data["url"],
            name=source_data["name"],
            config=config,
            status=SourceStatus.PENDING
        )
        
        try:
            self.db.add(db_source)
            self.db.commit()
            self.db.refresh(db_source)
            
            # Queue initial crawl job (if message queue available)
            try:
                from .message_queue import message_queue
                if message_queue and hasattr(message_queue, 'client') and message_queue.client:
                    await message_queue.publish(
                        "crawl_jobs",
                        json.dumps({
                            "job_type": "initial_crawl",
                            "source_id": str(db_source.id),
                            "url": db_source.url,
                            "config": db_source.config
                        })
                    )
                else:
                    logger.warning("Message queue not available, skipping job queue")
            except Exception as mq_error:
                logger.error(f"Failed to queue crawl job: {mq_error}")
                # Don't fail the entire operation if message queue fails
            
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
        
        # Convert update data to database format
        update_dict = update_data.to_db_update()
        
        # Handle config updates specially - merge with existing config
        if 'config' in update_dict:
            existing_config = source.config or {}
            # Merge new config with existing config
            existing_config.update(update_dict['config'])
            # Force SQLAlchemy to detect the change
            source.config = existing_config
            flag_modified(source, 'config')
            update_dict.pop('config')
        
        # Update other fields directly
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
    
    async def delete_source_data(self, source_id: UUID) -> bool:
        """Delete a knowledge source and all related data (for background processing)"""
        logger.info(f"Starting deletion process for source {source_id}")
        
        try:
            source = await self.get_by_id(source_id)
            if not source:
                logger.warning(f"Source {source_id} not found for deletion")
                return False
            
            # The database relationships have cascade="all, delete-orphan" 
            # so deleting the source will automatically delete:
            # - All associated documents
            # - All associated document chunks  
            # - All associated scraping jobs
            
            # Also need to clean up from vector store if integrated
            try:
                # This would integrate with vector store cleanup
                # For now just log the operation
                logger.info(f"Would clean up vector embeddings for source {source_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up vector store for source {source_id}: {e}")
                # Continue with database deletion even if vector store cleanup fails
            
            # Delete the source (cascades to related tables)
            self.db.delete(source)
            self.db.commit()
            
            logger.info(f"Successfully deleted source {source_id} and all related data")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting source {source_id}: {e}")
            self.db.rollback()
            return False