"""Memory model for storing extracted information"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from enum import Enum as PyEnum
from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Float, Integer, Enum
from sqlalchemy.dialects.postgresql import UUID as PGUUID, ARRAY, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property

from ...models import Base


class MemoryType(PyEnum):
    """Types of memories that can be extracted"""
    FACT = "fact"                # Factual information
    PREFERENCE = "preference"    # User preferences
    CODE = "code"               # Code snippets or patterns
    DECISION = "decision"       # Decisions made
    ERROR = "error"             # Errors encountered
    PATTERN = "pattern"         # Recognized patterns
    ENTITY = "entity"           # Entity information


class Memory(Base):
    """Model for storing extracted memories from conversations"""
    
    __tablename__ = 'memories'
    
    # Primary key
    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Session association
    session_id = Column(PGUUID(as_uuid=True),
                        ForeignKey('memory_sessions.id', ondelete='CASCADE'),
                        nullable=False, index=True,
                        comment='Session this memory belongs to')
    
    # Memory content
    content = Column(Text, nullable=False,
                     comment='The actual memory content')
    summary = Column(Text, nullable=True,
                     comment='Condensed version for quick reference')
    
    # Memory classification
    memory_type = Column(Enum('fact', 'preference', 'code', 'decision', 'error', 'pattern', 'entity', 
                              name='memory_type', create_type=False), 
                         nullable=False, index=True,
                         comment='Type of memory for categorization')
    
    # Importance and relevance
    importance = Column(Float, nullable=False, default=0.5, index=True,
                        comment='Importance score from 0.0 to 1.0')
    confidence = Column(Float, nullable=False, default=0.8,
                        comment='Confidence in extraction accuracy')
    
    # Entity and relationship tracking
    entities = Column(ARRAY(Text), nullable=True, default=list,
                      comment='Entities mentioned in this memory')
    related_memories = Column(ARRAY(PGUUID(as_uuid=True)), nullable=True, default=list,
                              comment='UUIDs of related memories')
    
    # Vector embedding for similarity search
    embedding = Column(ARRAY(Float), nullable=True,
                       comment='Vector embedding for semantic search')
    
    # Access tracking
    access_count = Column(Integer, nullable=False, default=0,
                          comment='Number of times this memory was accessed')
    last_accessed = Column(DateTime(timezone=True), nullable=True,
                           comment='Last time this memory was accessed')
    
    # Metadata
    memory_metadata = Column('metadata', JSONB, nullable=True, default=dict,
                             comment='Additional flexible metadata')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime(timezone=True), nullable=False,
                        default=lambda: datetime.now(timezone.utc), 
                        onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    session = relationship('MemorySession', back_populates='memories')
    
    # Properties
    @hybrid_property
    def age_days(self) -> float:
        """Calculate age of memory in days"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds() / 86400
    
    @hybrid_property
    def relevance_score(self) -> float:
        """Calculate current relevance score with time decay"""
        # Simple time decay function
        age_factor = 1.0 / (1.0 + self.age_days / 30)  # 30-day half-life
        access_factor = min(1.0, self.access_count / 10)  # Cap at 10 accesses
        return self.importance * age_factor * (0.7 + 0.3 * access_factor)
    
    @property
    def is_recent(self) -> bool:
        """Check if memory is recent (< 7 days old)"""
        return self.age_days < 7
    
    @property
    def is_high_importance(self) -> bool:
        """Check if memory has high importance"""
        return self.importance >= 0.7
    
    # Methods
    def update_access(self) -> None:
        """Update access count and timestamp"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
    
    def add_entity(self, entity: str) -> None:
        """Add an entity to this memory"""
        if self.entities is None:
            self.entities = []
        if entity not in self.entities:
            self.entities.append(entity)
    
    def add_related_memory(self, memory_id: UUID) -> None:
        """Add a related memory reference"""
        if self.related_memories is None:
            self.related_memories = []
        if memory_id not in self.related_memories:
            self.related_memories.append(memory_id)
    
    def set_embedding(self, embedding: List[float]) -> None:
        """Set the vector embedding"""
        self.embedding = embedding
    
    def to_context_string(self) -> str:
        """Convert to a string suitable for context injection"""
        type_prefix = {
            "fact": "📌 Fact:",
            "preference": "⚙️ Preference:",
            "code": "💻 Code:",
            "decision": "🎯 Decision:",
            "error": "❌ Error:",
            "pattern": "🔄 Pattern:",
            "entity": "🏷️ Entity:"
        }
        
        prefix = type_prefix.get(self.memory_type, "📝 Memory:")
        content = self.summary or self.content
        
        if self.entities:
            entities_str = f" [Entities: {', '.join(self.entities)}]"
        else:
            entities_str = ""
        
        return f"{prefix} {content}{entities_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'id': str(self.id),
            'session_id': str(self.session_id),
            'content': self.content,
            'summary': self.summary,
            'memory_type': self.memory_type,
            'importance': self.importance,
            'confidence': self.confidence,
            'entities': self.entities or [],
            'related_memories': [str(m) for m in (self.related_memories or [])],
            'has_embedding': self.embedding is not None,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'metadata': self.memory_metadata or {},
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'age_days': self.age_days,
            'relevance_score': self.relevance_score,
            'is_recent': self.is_recent,
            'is_high_importance': self.is_high_importance
        }
    
    def __repr__(self) -> str:
        return (f"<Memory(id={self.id}, type={self.memory_type}, "
                f"importance={self.importance}, session={self.session_id})>")