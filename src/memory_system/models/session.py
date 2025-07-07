"""Session model for memory system"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property

from ...api.models import Base


class MemorySession(Base):
    """Model for tracking Claude-Code conversation sessions"""
    
    __tablename__ = 'memory_sessions'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # User and project identification
    user_id = Column(String(255), nullable=False, index=True,
                     comment='Unique identifier for the user')
    project_id = Column(PGUUID(as_uuid=True), nullable=True, index=True,
                        comment='Optional project association')
    
    # Session lifecycle
    started_at = Column(DateTime(timezone=True), nullable=False,
                        default=datetime.utcnow, index=True,
                        comment='When the session started')
    ended_at = Column(DateTime(timezone=True), nullable=True,
                      comment='When the session ended')
    
    # Session metadata
    metadata = Column(JSONB, nullable=True, default=dict,
                      comment='Flexible metadata storage')
    tags = Column(ARRAY(Text), nullable=True, default=list,
                  comment='Session tags for categorization')
    
    # Session linking
    parent_session_id = Column(PGUUID(as_uuid=True), 
                               ForeignKey('memory_sessions.id', ondelete='SET NULL'),
                               nullable=True, index=True,
                               comment='Link to parent session for continuity')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False,
                        default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    memories = relationship('Memory', back_populates='session',
                            cascade='all, delete-orphan',
                            order_by='Memory.created_at.desc()')
    parent_session = relationship('MemorySession', remote_side=[id],
                                  backref=backref('child_sessions'))
    
    # Properties
    @hybrid_property
    def duration(self) -> Optional[float]:
        """Calculate session duration in seconds"""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None
    
    @hybrid_property
    def is_active(self) -> bool:
        """Check if session is still active"""
        return self.ended_at is None
    
    @property
    def memory_count(self) -> int:
        """Get count of memories in this session"""
        return len(self.memories)
    
    @property
    def important_memories(self) -> List['Memory']:
        """Get memories with high importance (>= 0.7)"""
        return [m for m in self.memories if m.importance >= 0.7]
    
    # Methods
    def end_session(self) -> None:
        """Mark session as ended"""
        if not self.ended_at:
            self.ended_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the session"""
        if self.tags is None:
            self.tags = []
        if tag not in self.tags:
            self.tags.append(tag)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add or update metadata"""
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of session context"""
        return {
            'session_id': str(self.id),
            'user_id': self.user_id,
            'project_id': str(self.project_id) if self.project_id else None,
            'started_at': self.started_at.isoformat(),
            'duration': self.duration,
            'memory_count': self.memory_count,
            'important_memories': len(self.important_memories),
            'tags': self.tags or [],
            'is_active': self.is_active
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'user_id': self.user_id,
            'project_id': str(self.project_id) if self.project_id else None,
            'started_at': self.started_at.isoformat(),
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'metadata': self.metadata or {},
            'tags': self.tags or [],
            'parent_session_id': str(self.parent_session_id) if self.parent_session_id else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'duration': self.duration,
            'is_active': self.is_active,
            'memory_count': self.memory_count
        }
    
    def __repr__(self) -> str:
        return (f"<MemorySession(id={self.id}, user={self.user_id}, "
                f"started={self.started_at}, active={self.is_active})>")