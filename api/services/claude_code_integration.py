"""
Claude Code Integration Module
Provides bidirectional memory sync, session continuity, and real-time context sharing
"""

import asyncio
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import uuid

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models import get_db
from ..models.memory import MemoryItem
from ..services.realtime_learning_pipeline import (
    get_learning_pipeline,
    StreamEvent,
    EventType
)
from ..services.pattern_recognition_engine import get_pattern_engine
from ..services.memory_service import MemoryService
# from ..services.claude_session_manager import ClaudeSessionManager
from ..schemas.memory import MemoryCreate

logger = logging.getLogger(__name__)


class ClaudeContext(BaseModel):
    """Represents Claude's current context"""
    session_id: str
    workspace_path: str
    current_files: List[str] = Field(default_factory=list)
    recent_edits: List[Dict[str, Any]] = Field(default_factory=list)
    active_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    learned_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemorySyncRequest(BaseModel):
    """Request for memory synchronization"""
    source_session_id: str
    target_session_id: Optional[str] = None
    sync_type: str = "full"  # full, incremental, selective
    filters: Dict[str, Any] = Field(default_factory=dict)


class ContextUpdateRequest(BaseModel):
    """Request to update Claude's context"""
    session_id: str
    update_type: str  # file_change, task_update, error_occurred, decision_made
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaudeCodeIntegration:
    """
    Core integration module for Claude Code that provides:
    - Bidirectional memory synchronization
    - Session continuity across instances
    - Real-time context sharing
    - Intelligent context handoff
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, ClaudeContext] = {}
        self.session_links: Dict[str, Set[str]] = defaultdict(set)
        self.memory_service = MemoryService()
        # self.session_manager = ClaudeSessionManager()
        self.sync_locks: Dict[str, asyncio.Lock] = {}
        
    async def initialize_session(
        self,
        workspace_path: str,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        db: Session = None
    ) -> Dict[str, Any]:
        """Initialize a new Claude Code session with context restoration"""
        
        # Create new session ID
        session_id = str(uuid.uuid4())
        
        # Initialize context
        context = ClaudeContext(
            session_id=session_id,
            workspace_path=workspace_path
        )
        
        # If parent session exists, inherit context
        if parent_session_id and parent_session_id in self.active_sessions:
            parent_context = self.active_sessions[parent_session_id]
            context.learned_patterns = parent_context.learned_patterns.copy()
            context.error_history = parent_context.error_history[-10:]  # Keep recent errors
            context.decisions_made = parent_context.decisions_made[-20:]  # Keep recent decisions
            
            # Link sessions for bidirectional sync
            self.session_links[session_id].add(parent_session_id)
            self.session_links[parent_session_id].add(session_id)
            
        # Store active session
        self.active_sessions[session_id] = context
        self.sync_locks[session_id] = asyncio.Lock()
        
        # Restore memories from database
        restored_data = await self._restore_session_memories(workspace_path, user_id, db)
        
        # Get learning pipeline for real-time updates
        pipeline = await get_learning_pipeline()
        
        # Publish session start event
        await pipeline.publish_event(StreamEvent(
            event_type=EventType.SESSION_EVENT,
            session_id=session_id,
            user_id=user_id,
            data={
                "event": "session_started",
                "workspace": workspace_path,
                "parent_session": parent_session_id,
                "restored_memories": len(restored_data.get("memories", []))
            }
        ))
        
        return {
            "session_id": session_id,
            "status": "initialized",
            "context": context.dict(),
            "restored": restored_data,
            "linked_sessions": list(self.session_links.get(session_id, set()))
        }
        
    async def sync_memories(
        self,
        sync_request: MemorySyncRequest,
        db: Session
    ) -> Dict[str, Any]:
        """Synchronize memories between Claude sessions"""
        
        source_id = sync_request.source_session_id
        target_id = sync_request.target_session_id
        
        # Get sync lock
        async with self.sync_locks.get(source_id, asyncio.Lock()):
            
            # Get source context
            source_context = self.active_sessions.get(source_id)
            if not source_context:
                raise ValueError(f"Source session {source_id} not found")
                
            # Determine target sessions
            if target_id:
                target_sessions = [target_id]
            else:
                # Sync to all linked sessions
                target_sessions = list(self.session_links.get(source_id, set()))
                
            sync_results = []
            
            for target in target_sessions:
                if target not in self.active_sessions:
                    continue
                    
                target_context = self.active_sessions[target]
                
                # Perform sync based on type
                if sync_request.sync_type == "full":
                    # Full sync - copy everything
                    target_context.learned_patterns = source_context.learned_patterns.copy()
                    target_context.error_history = source_context.error_history.copy()
                    target_context.decisions_made = source_context.decisions_made.copy()
                    
                elif sync_request.sync_type == "incremental":
                    # Incremental sync - only new items
                    await self._incremental_sync(source_context, target_context, sync_request.filters)
                    
                elif sync_request.sync_type == "selective":
                    # Selective sync based on filters
                    await self._selective_sync(source_context, target_context, sync_request.filters)
                    
                sync_results.append({
                    "target_session": target,
                    "items_synced": {
                        "patterns": len(target_context.learned_patterns),
                        "errors": len(target_context.error_history),
                        "decisions": len(target_context.decisions_made)
                    }
                })
                
            # Persist to database
            await self._persist_sync_state(source_id, target_sessions, db)
            
            # Publish sync event
            pipeline = await get_learning_pipeline()
            await pipeline.publish_event(StreamEvent(
                event_type=EventType.CONTEXT_UPDATED,
                session_id=source_id,
                data={
                    "sync_type": sync_request.sync_type,
                    "targets": target_sessions,
                    "results": sync_results
                }
            ))
            
            return {
                "source": source_id,
                "sync_results": sync_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def update_context(
        self,
        update_request: ContextUpdateRequest
    ) -> Dict[str, Any]:
        """Update Claude's context in real-time"""
        
        session_id = update_request.session_id
        context = self.active_sessions.get(session_id)
        
        if not context:
            raise ValueError(f"Session {session_id} not found")
            
        # Update based on type
        if update_request.update_type == "file_change":
            file_path = update_request.data.get("file_path")
            if file_path and file_path not in context.current_files:
                context.current_files.append(file_path)
            context.recent_edits.append({
                "file": file_path,
                "changes": update_request.data.get("changes", []),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Analyze for patterns
            if update_request.data.get("code"):
                await self._analyze_code_patterns(
                    update_request.data["code"],
                    session_id,
                    file_path
                )
                
        elif update_request.update_type == "task_update":
            task = update_request.data
            # Update or add task
            existing = next((t for t in context.active_tasks if t.get("id") == task.get("id")), None)
            if existing:
                existing.update(task)
            else:
                context.active_tasks.append(task)
                
        elif update_request.update_type == "error_occurred":
            error_info = {
                **update_request.data,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id
            }
            context.error_history.append(error_info)
            
            # Learn from error
            await self._learn_from_error(error_info, session_id)
            
        elif update_request.update_type == "decision_made":
            decision = {
                **update_request.data,
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id
            }
            context.decisions_made.append(decision)
            
        # Propagate updates to linked sessions
        await self._propagate_context_update(session_id, update_request)
        
        # Publish context update event
        pipeline = await get_learning_pipeline()
        await pipeline.publish_event(StreamEvent(
            event_type=EventType.CONTEXT_UPDATED,
            session_id=session_id,
            data={
                "update_type": update_request.update_type,
                "data": update_request.data
            }
        ))
        
        return {
            "session_id": session_id,
            "update_type": update_request.update_type,
            "context_summary": self._summarize_context(context),
            "propagated_to": list(self.session_links.get(session_id, set()))
        }
        
    async def get_unified_context(
        self,
        session_id: str,
        include_linked: bool = True
    ) -> Dict[str, Any]:
        """Get unified context across linked sessions"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
            
        contexts = [self.active_sessions[session_id]]
        
        if include_linked:
            for linked_id in self.session_links.get(session_id, set()):
                if linked_id in self.active_sessions:
                    contexts.append(self.active_sessions[linked_id])
                    
        # Merge contexts
        unified = {
            "primary_session": session_id,
            "linked_sessions": list(self.session_links.get(session_id, set())),
            "workspace_paths": list(set(c.workspace_path for c in contexts)),
            "all_files": list(set(f for c in contexts for f in c.current_files)),
            "combined_patterns": self._merge_patterns(contexts),
            "error_insights": self._analyze_error_patterns(contexts),
            "decision_history": self._merge_decisions(contexts),
            "active_tasks": self._merge_tasks(contexts),
            "context_age": min(c.timestamp for c in contexts).isoformat(),
            "last_update": max(c.timestamp for c in contexts).isoformat()
        }
        
        return unified
        
    async def handoff_session(
        self,
        current_session_id: str,
        notes: str,
        next_tasks: List[str],
        db: Session
    ) -> Dict[str, Any]:
        """Create handoff for next Claude session"""
        
        context = self.active_sessions.get(current_session_id)
        if not context:
            raise ValueError(f"Session {current_session_id} not found")
            
        # Create handoff memory
        handoff_data = {
            "session_id": current_session_id,
            "workspace": context.workspace_path,
            "notes": notes,
            "next_tasks": next_tasks,
            "context_summary": self._summarize_context(context),
            "recent_patterns": context.learned_patterns[-5:],
            "recent_errors": context.error_history[-5:],
            "recent_decisions": context.decisions_made[-10:],
            "active_files": context.current_files,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save as memory
        memory_data = MemoryCreate(
            content=json.dumps(handoff_data),
            tags=["handoff", f"session:{current_session_id}", f"workspace:{context.workspace_path}"],
            metadata={
                "type": "handoff",
                "session_id": current_session_id,
                "workspace": context.workspace_path,
                "importance": 0.9
            }
        )
        memory = await self.memory_service.create_memory(db, memory_data)
        
        # Clean up session (but keep in links for future reference)
        if current_session_id in self.active_sessions:
            # Archive context
            await self._archive_session_context(current_session_id, db)
            
        return {
            "handoff_id": memory.id,
            "session_id": current_session_id,
            "handoff_data": handoff_data,
            "status": "completed"
        }
        
    async def _restore_session_memories(
        self,
        workspace_path: str,
        user_id: Optional[str],
        db: Session
    ) -> Dict[str, Any]:
        """Restore relevant memories for a session"""
        
        # Search for memories related to this workspace
        search_results = await self.memory_service.search_memories(
            db=db,
            query_text=f"workspace:{workspace_path}",
            limit=50
        )
        
        memories = []
        handoffs = []
        learnings = []
        
        # Process search results
        for result in search_results:
            memory = result.get("memory")
            if memory:
                memories.append(memory.to_dict())
                
                # Check tags for memory type
                if "handoff" in memory.tags:
                    handoffs.append(memory)
                elif "learning" in memory.tags:
                    learnings.append(memory)
        
        return {
            "memories": memories,
            "handoffs": handoffs,
            "learnings": learnings,
            "total_restored": len(memories)
        }
        
    async def _analyze_code_patterns(
        self,
        code: str,
        session_id: str,
        file_path: str
    ):
        """Analyze code for patterns and update context"""
        
        pattern_engine = get_pattern_engine()
        patterns = await pattern_engine.analyze_code(code, "python")
        
        if patterns:
            context = self.active_sessions.get(session_id)
            if context:
                for pattern in patterns:
                    pattern_data = {
                        "pattern": pattern.name,
                        "type": pattern.pattern_type,
                        "file": file_path,
                        "confidence": pattern.confidence,
                        "detected_at": datetime.utcnow().isoformat()
                    }
                    context.learned_patterns.append(pattern_data)
                    
    async def _learn_from_error(
        self,
        error_info: Dict[str, Any],
        session_id: str
    ):
        """Learn from errors to prevent future occurrences"""
        
        # Get similar errors from history
        context = self.active_sessions.get(session_id)
        if not context:
            return
            
        similar_errors = [
            e for e in context.error_history
            if e.get("error_type") == error_info.get("error_type")
        ]
        
        if len(similar_errors) > 2:
            # Recurring error - create learning
            learning = {
                "type": "recurring_error",
                "error_type": error_info.get("error_type"),
                "occurrences": len(similar_errors),
                "pattern": "Consider refactoring to prevent this error",
                "learned_at": datetime.utcnow().isoformat()
            }
            context.learned_patterns.append(learning)
            
    async def _propagate_context_update(
        self,
        source_session_id: str,
        update_request: ContextUpdateRequest
    ):
        """Propagate context updates to linked sessions"""
        
        for linked_id in self.session_links.get(source_session_id, set()):
            if linked_id in self.active_sessions:
                # Apply update to linked session
                linked_request = ContextUpdateRequest(
                    session_id=linked_id,
                    update_type=update_request.update_type,
                    data=update_request.data,
                    metadata={
                        **update_request.metadata,
                        "propagated_from": source_session_id
                    }
                )
                # Recursive call but won't propagate again due to metadata
                if not linked_request.metadata.get("propagated_from"):
                    await self.update_context(linked_request)
                    
    async def _incremental_sync(
        self,
        source: ClaudeContext,
        target: ClaudeContext,
        filters: Dict[str, Any]
    ):
        """Perform incremental sync of new items only"""
        
        # Sync only items newer than target's last update
        cutoff = target.timestamp
        
        # Sync new patterns
        new_patterns = [
            p for p in source.learned_patterns
            if datetime.fromisoformat(p.get("detected_at", "2020-01-01")) > cutoff
        ]
        target.learned_patterns.extend(new_patterns)
        
        # Sync new errors
        new_errors = [
            e for e in source.error_history
            if datetime.fromisoformat(e.get("timestamp", "2020-01-01")) > cutoff
        ]
        target.error_history.extend(new_errors)
        
        # Update timestamp
        target.timestamp = datetime.utcnow()
        
    async def _selective_sync(
        self,
        source: ClaudeContext,
        target: ClaudeContext,
        filters: Dict[str, Any]
    ):
        """Perform selective sync based on filters"""
        
        # Filter patterns by type
        if pattern_types := filters.get("pattern_types"):
            patterns = [
                p for p in source.learned_patterns
                if p.get("type") in pattern_types
            ]
            target.learned_patterns = patterns
            
        # Filter errors by type
        if error_types := filters.get("error_types"):
            errors = [
                e for e in source.error_history
                if e.get("error_type") in error_types
            ]
            target.error_history = errors
            
    async def _persist_sync_state(
        self,
        source_id: str,
        target_ids: List[str],
        db: Session
    ):
        """Persist synchronization state to database"""
        
        sync_memory = {
            "type": "sync_event",
            "source": source_id,
            "targets": target_ids,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        memory_data = MemoryCreate(
            content=json.dumps(sync_memory),
            tags=["system", "sync_event", f"source:{source_id}"],
            metadata={
                "type": "system",
                "event": "context_sync",
                "source": source_id,
                "targets": target_ids
            }
        )
        await self.memory_service.create_memory(db, memory_data)
        
    async def _archive_session_context(
        self,
        session_id: str,
        db: Session
    ):
        """Archive session context when ending"""
        
        context = self.active_sessions.get(session_id)
        if not context:
            return
            
        archive_data = {
            "session_id": session_id,
            "workspace": context.workspace_path,
            "duration": (datetime.utcnow() - context.timestamp).total_seconds(),
            "files_touched": len(context.current_files),
            "patterns_learned": len(context.learned_patterns),
            "errors_encountered": len(context.error_history),
            "decisions_made": len(context.decisions_made),
            "final_state": {
                "session_id": context.session_id,
                "workspace_path": context.workspace_path,
                "current_files": context.current_files,
                "recent_edits": context.recent_edits,
                "active_tasks": context.active_tasks,
                "learned_patterns": context.learned_patterns,
                "error_history": context.error_history,
                "decisions_made": context.decisions_made,
                "timestamp": context.timestamp.isoformat()
            }
        }
        
        memory_data = MemoryCreate(
            content=json.dumps(archive_data),
            tags=["session_archive", f"session:{session_id}", f"workspace:{context.workspace_path}"],
            metadata={
                "type": "session_archive",
                "session_id": session_id,
                "workspace": context.workspace_path
            }
        )
        await self.memory_service.create_memory(db, memory_data)
        
    def _summarize_context(self, context: ClaudeContext) -> Dict[str, Any]:
        """Create a summary of the current context"""
        
        return {
            "files_active": len(context.current_files),
            "recent_edits": len(context.recent_edits),
            "patterns_learned": len(context.learned_patterns),
            "errors_encountered": len(context.error_history),
            "decisions_made": len(context.decisions_made),
            "tasks_active": len([t for t in context.active_tasks if t.get("status") != "completed"]),
            "session_age": (datetime.utcnow() - context.timestamp).total_seconds()
        }
        
    def _merge_patterns(self, contexts: List[ClaudeContext]) -> List[Dict[str, Any]]:
        """Merge patterns from multiple contexts"""
        
        all_patterns = []
        seen = set()
        
        for context in contexts:
            for pattern in context.learned_patterns:
                # Create unique key
                key = f"{pattern.get('pattern')}_{pattern.get('type')}_{pattern.get('file')}"
                if key not in seen:
                    seen.add(key)
                    all_patterns.append(pattern)
                    
        return sorted(all_patterns, key=lambda p: p.get("confidence", 0), reverse=True)
        
    def _analyze_error_patterns(self, contexts: List[ClaudeContext]) -> Dict[str, Any]:
        """Analyze error patterns across contexts"""
        
        error_types = defaultdict(int)
        error_files = defaultdict(int)
        
        for context in contexts:
            for error in context.error_history:
                error_types[error.get("error_type", "unknown")] += 1
                if file_path := error.get("file"):
                    error_files[file_path] += 1
                    
        return {
            "most_common_errors": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
            "problematic_files": sorted(error_files.items(), key=lambda x: x[1], reverse=True)[:5],
            "total_errors": sum(error_types.values())
        }
        
    def _merge_decisions(self, contexts: List[ClaudeContext]) -> List[Dict[str, Any]]:
        """Merge decisions from multiple contexts"""
        
        all_decisions = []
        for context in contexts:
            all_decisions.extend(context.decisions_made)
            
        # Sort by timestamp, most recent first
        return sorted(
            all_decisions,
            key=lambda d: d.get("timestamp", ""),
            reverse=True
        )[:50]  # Keep top 50
        
    def _merge_tasks(self, contexts: List[ClaudeContext]) -> List[Dict[str, Any]]:
        """Merge active tasks from multiple contexts"""
        
        task_map = {}
        
        for context in contexts:
            for task in context.active_tasks:
                task_id = task.get("id")
                if task_id:
                    # If task exists, take the most recent version
                    if task_id not in task_map or task.get("updated_at", "") > task_map[task_id].get("updated_at", ""):
                        task_map[task_id] = task
                else:
                    # Tasks without IDs are always included
                    task_map[str(uuid.uuid4())] = task
                    
        return list(task_map.values())


# Singleton instance
_integration_instance: Optional[ClaudeCodeIntegration] = None


def get_claude_integration() -> ClaudeCodeIntegration:
    """Get or create the Claude Code integration instance"""
    global _integration_instance
    
    if _integration_instance is None:
        _integration_instance = ClaudeCodeIntegration()
        
    return _integration_instance