"""
RAG (Retrieval-Augmented Generation) services
"""

# Export the simple service by default
from .simple_rag_service import SimpleRAGService, get_rag_service

# Try to export the full service if available
try:
    from .llamaindex_service import LlamaIndexRAGService
except ImportError:
    LlamaIndexRAGService = None

__all__ = [
    "SimpleRAGService",
    "get_rag_service",
    "LlamaIndexRAGService"
]