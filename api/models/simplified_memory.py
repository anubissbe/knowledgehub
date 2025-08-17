
from enum import Enum
from typing import Dict, Any, Optional

class SimplifiedMemoryType(str, Enum):
    """Simplified memory type system"""
    # Core Types (5 instead of 55)
    CONVERSATION = "conversation"  # Dialog and chat history
    KNOWLEDGE = "knowledge"        # Facts and information
    TASK = "task"                 # Tasks and actions
    CONTEXT = "context"            # Environmental context
    SYSTEM = "system"              # System state and config

class MemoryCategory(str, Enum):
    """Memory categories for sub-classification"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryTypeMapper:
    """Maps old complex types to simplified system"""
    
    LEGACY_MAPPING = {
        # Conversation types
        "conversation": SimplifiedMemoryType.CONVERSATION,
        "dialog": SimplifiedMemoryType.CONVERSATION,
        "chat": SimplifiedMemoryType.CONVERSATION,
        "message": SimplifiedMemoryType.CONVERSATION,
        
        # Knowledge types
        "general": SimplifiedMemoryType.KNOWLEDGE,
        "fact": SimplifiedMemoryType.KNOWLEDGE,
        "concept": SimplifiedMemoryType.KNOWLEDGE,
        "definition": SimplifiedMemoryType.KNOWLEDGE,
        
        # Task types
        "task": SimplifiedMemoryType.TASK,
        "action": SimplifiedMemoryType.TASK,
        "plan": SimplifiedMemoryType.TASK,
        "goal": SimplifiedMemoryType.TASK,
        
        # Context types
        "context": SimplifiedMemoryType.CONTEXT,
        "environment": SimplifiedMemoryType.CONTEXT,
        "situation": SimplifiedMemoryType.CONTEXT,
        
        # System types
        "system": SimplifiedMemoryType.SYSTEM,
        "config": SimplifiedMemoryType.SYSTEM,
        "state": SimplifiedMemoryType.SYSTEM,
    }
    
    @classmethod
    def map_legacy_type(cls, legacy_type: str) -> SimplifiedMemoryType:
        """Map legacy type to simplified type"""
        # Check direct mapping
        if legacy_type in cls.LEGACY_MAPPING:
            return cls.LEGACY_MAPPING[legacy_type]
        
        # Default based on keywords
        legacy_lower = legacy_type.lower()
        if any(word in legacy_lower for word in ["chat", "dialog", "message"]):
            return SimplifiedMemoryType.CONVERSATION
        elif any(word in legacy_lower for word in ["fact", "know", "info"]):
            return SimplifiedMemoryType.KNOWLEDGE
        elif any(word in legacy_lower for word in ["task", "action", "do"]):
            return SimplifiedMemoryType.TASK
        elif any(word in legacy_lower for word in ["context", "environment"]):
            return SimplifiedMemoryType.CONTEXT
        else:
            return SimplifiedMemoryType.SYSTEM
