"""Source schemas"""

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

from ..security import InputSanitizer


class SourceCreate(BaseModel):
    """Schema for creating a new knowledge source"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    url: HttpUrl
    type: Optional[str] = Field(default="website")
    refresh_interval: Optional[int] = Field(default=86400)  # 24 hours in seconds
    authentication: Optional[Dict[str, Any]] = None
    crawl_config: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})
    
    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize source name to prevent XSS"""
        if v is not None:
            return InputSanitizer.sanitize_text(v, max_length=255, allow_html=False)
        return v
    
    @validator('url')
    def validate_url(cls, v):
        """Validate and sanitize URL"""
        url_str = str(v)
        try:
            sanitized_url = InputSanitizer.sanitize_url(url_str)
            return sanitized_url
        except ValueError as e:
            raise ValueError(f"Invalid URL: {e}")
    
    @validator('type')
    def sanitize_type(cls, v):
        """Sanitize source type"""
        if v is not None:
            # Allow only specific source types
            allowed_types = ['website', 'documentation', 'wiki', 'repository', 'api']
            sanitized = InputSanitizer.sanitize_text(v, max_length=50, allow_html=False)
            if sanitized.lower() not in allowed_types:
                raise ValueError(f"Invalid source type. Allowed: {allowed_types}")
            return sanitized.lower()
        return v
    
    @validator('authentication')
    def sanitize_authentication(cls, v):
        """Sanitize authentication configuration"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v
    
    @validator('crawl_config')
    def sanitize_crawl_config(cls, v):
        """Sanitize crawl configuration"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v
    
    @validator('config')
    def sanitize_config(cls, v):
        """Sanitize general configuration"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v
    
    def to_db_config(self) -> Dict[str, Any]:
        """Convert frontend fields to backend config format"""
        config = self.config.copy() if self.config else {}
        config.update({
            "type": self.type,
            "refresh_interval": self.refresh_interval,
        })
        if self.authentication:
            config["authentication"] = self.authentication
        if self.crawl_config:
            config["crawl_config"] = self.crawl_config
        return config


class SourceUpdate(BaseModel):
    """Schema for updating a knowledge source"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    type: Optional[str] = None
    refresh_interval: Optional[int] = None
    authentication: Optional[Dict[str, Any]] = None
    crawl_config: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    
    @validator('name')
    def sanitize_name(cls, v):
        """Sanitize source name to prevent XSS"""
        if v is not None:
            return InputSanitizer.sanitize_text(v, max_length=255, allow_html=False)
        return v
    
    @validator('type')
    def sanitize_type(cls, v):
        """Sanitize source type"""
        if v is not None:
            # Allow only specific source types
            allowed_types = ['website', 'documentation', 'wiki', 'repository', 'api']
            sanitized = InputSanitizer.sanitize_text(v, max_length=50, allow_html=False)
            if sanitized.lower() not in allowed_types:
                raise ValueError(f"Invalid source type. Allowed: {allowed_types}")
            return sanitized.lower()
        return v
    
    @validator('authentication')
    def sanitize_authentication(cls, v):
        """Sanitize authentication configuration"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v
    
    @validator('crawl_config')
    def sanitize_crawl_config(cls, v):
        """Sanitize crawl configuration"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v
    
    @validator('config')
    def sanitize_config(cls, v):
        """Sanitize general configuration"""
        if v is not None:
            return InputSanitizer.sanitize_dict(v)
        return v
    
    def to_db_update(self) -> Dict[str, Any]:
        """Convert update fields to database format"""
        update_dict = self.dict(exclude_unset=True)
        
        # If individual fields are being updated, merge them into config
        if any(key in update_dict for key in ['type', 'refresh_interval', 'authentication', 'crawl_config']):
            # Get existing config or create new one
            config = update_dict.get('config', {})
            
            # Merge individual fields into config
            if 'type' in update_dict:
                config['type'] = update_dict.pop('type')
            if 'refresh_interval' in update_dict:
                config['refresh_interval'] = update_dict.pop('refresh_interval')
            if 'authentication' in update_dict:
                config['authentication'] = update_dict.pop('authentication')
            if 'crawl_config' in update_dict:
                config['crawl_config'] = update_dict.pop('crawl_config')
            
            update_dict['config'] = config
        
        return update_dict


class SourceResponse(BaseModel):
    """Schema for knowledge source response"""
    id: UUID
    name: str
    url: str
    status: str
    type: str
    refresh_interval: int
    authentication: Optional[Dict[str, Any]] = None
    crawl_config: Optional[Dict[str, Any]] = None
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_scraped_at: Optional[datetime]
    stats: Dict[str, Any]
    
    @classmethod
    def from_db_model(cls, source):
        """Create response from database model, extracting config fields"""
        config = source.config or {}
        return cls(
            id=source.id,
            name=source.name,
            url=source.url,
            status=source.status.value if hasattr(source.status, 'value') else source.status,
            type=config.get("type", "website"),
            refresh_interval=config.get("refresh_interval", 86400),
            authentication=config.get("authentication"),
            crawl_config=config.get("crawl_config"),
            config=config,
            created_at=source.created_at,
            updated_at=source.updated_at,
            last_scraped_at=source.last_scraped_at,
            stats=source.stats or {}
        )
    
    class Config:
        from_attributes = True


class SourceListResponse(BaseModel):
    """Schema for list of sources response"""
    sources: List[SourceResponse]
    total: int
    skip: int
    limit: int