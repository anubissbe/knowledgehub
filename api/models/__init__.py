"""Database models for AI Knowledge Hub"""

from .base import Base, get_db, init_db
from .knowledge_source import KnowledgeSource
from .document import Document, DocumentChunk
from .job import ScrapingJob
from .memory import MemoryItem
from .auth import APIKey
from .search import SearchHistory
from .mistake_tracking import MistakeTracking, ErrorPattern
from .session import Session, SessionHandoff, SessionCheckpoint, SessionMetrics
from .error_pattern import EnhancedErrorPattern, ErrorOccurrence, ErrorSolution, ErrorFeedback, ErrorPrediction
from .task import Task, TaskStatus, TaskPriority

__all__ = [
    "Base",
    "get_db",
    "init_db",
    "KnowledgeSource",
    "Document",
    "DocumentChunk",
    "ScrapingJob",
    "MemoryItem",
    "APIKey",
    "SearchHistory",
    "MistakeTracking",
    "ErrorPattern",
    "Session",
    "SessionHandoff",
    "SessionCheckpoint",
    "SessionMetrics",
    "EnhancedErrorPattern",
    "ErrorOccurrence",
    "ErrorSolution",
    "ErrorFeedback",
    "ErrorPrediction",
    "Task",
    "TaskStatus",
    "TaskPriority",
]