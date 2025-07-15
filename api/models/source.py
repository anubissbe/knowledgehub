"""Source model - using knowledge_source.py"""

# Re-export from knowledge_source
from .knowledge_source import KnowledgeSource

# Alias for backward compatibility
Source = KnowledgeSource

__all__ = ["Source", "KnowledgeSource"]