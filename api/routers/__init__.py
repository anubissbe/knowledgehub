"""API routers"""

from . import sources, search, jobs, websocket, memories, chunks, documents, scheduler, exports

# Try to import workflow integration
try:
    from . import workflow_integration
    __all__ = ["sources", "search", "jobs", "websocket", "memories", "chunks", "documents", "scheduler", "exports", "workflow_integration"]
except ImportError:
    __all__ = ["sources", "search", "jobs", "websocket", "memories", "chunks", "documents", "scheduler", "exports"]