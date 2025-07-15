"""
Session Continuity Service for Claude Code
Enables linking sessions and maintaining context across conversations
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import json
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from ..models.memory import MemoryItem
from ..schemas.memory import MemoryCreate, MemoryResponse
from .memory_service import MemoryService


class SessionContinuityService:
    """Manages session linking and context continuity for Claude Code"""
    
    def __init__(self, db: Session, memory_service: MemoryService):
        self.db = db
        self.memory_service = memory_service
        
    async def continue_session(self, previous_session_id: str, user_id: str = "claude-code") -> Dict[str, Any]:
        """
        Continue from a previous session, loading relevant context
        
        Args:
            previous_session_id: The session ID to continue from
            user_id: The user ID (defaults to claude-code)
            
        Returns:
            Dict containing new session info and loaded context
        """
        # Get previous session memories
        previous_memories = await self.memory_service.get_session_memories(
            session_id=previous_session_id,
            user_id=user_id
        )
        
        # Create new session with link to previous
        new_session = await self.memory_service.create_session(
            user_id=user_id,
            metadata={
                "previous_session_id": previous_session_id,
                "session_type": "continuation",
                "started_at": datetime.utcnow().isoformat()
            }
        )
        
        # Create handoff memory
        handoff_content = self._create_handoff_summary(previous_memories)
        await self.memory_service.create_memory(
            session_id=new_session["id"],
            content=f"SESSION CONTINUATION: Continuing from session {previous_session_id}. {handoff_content}",
            memory_type="decision",
            importance=0.9,
            metadata={
                "session_link": previous_session_id,
                "handoff_type": "automatic"
            }
        )
        
        # Get relevant context
        context = await self.get_relevant_context(
            new_session["id"], 
            previous_session_id,
            user_id
        )
        
        return {
            "session_id": new_session["id"],
            "previous_session_id": previous_session_id,
            "context": context,
            "handoff_summary": handoff_content
        }
    
    async def get_relevant_context(
        self, 
        current_session_id: str,
        previous_session_id: Optional[str] = None,
        user_id: str = "claude-code",
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Get relevant context for current session
        
        Includes:
        - Recent memories from previous session
        - High-importance memories
        - Unfinished tasks
        - Recent errors and their solutions
        """
        context_parts = []
        
        # 1. Get session handoff notes
        handoff_memories = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"handoff_type": "manual"}),
                Memory.session_id == previous_session_id if previous_session_id else True
            )
        ).order_by(desc(Memory.created_at)).limit(5).all()
        
        if handoff_memories:
            context_parts.append({
                "type": "handoff_notes",
                "memories": [self._memory_to_dict(m) for m in handoff_memories]
            })
        
        # 2. Get unfinished tasks
        task_memories = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"task_status": "in_progress"}),
                Memory.created_at > datetime.utcnow() - timedelta(days=7)
            )
        ).order_by(desc(Memory.importance)).limit(10).all()
        
        if task_memories:
            context_parts.append({
                "type": "unfinished_tasks", 
                "memories": [self._memory_to_dict(m) for m in task_memories]
            })
        
        # 3. Get recent high-importance memories
        important_memories = self.db.query(Memory).filter(
            and_(
                Memory.importance >= 0.8,
                Memory.memory_type.in_(["decision", "error", "pattern"]),
                Memory.created_at > datetime.utcnow() - timedelta(days=30)
            )
        ).order_by(desc(Memory.importance), desc(Memory.created_at)).limit(10).all()
        
        if important_memories:
            context_parts.append({
                "type": "important_context",
                "memories": [self._memory_to_dict(m) for m in important_memories]
            })
        
        # 4. Get recent errors and solutions
        error_memories = self.db.query(Memory).filter(
            and_(
                Memory.memory_type == "error",
                Memory.created_at > datetime.utcnow() - timedelta(days=14)
            )
        ).order_by(desc(Memory.created_at)).limit(5).all()
        
        if error_memories:
            context_parts.append({
                "type": "recent_errors",
                "memories": [self._memory_to_dict(m) for m in error_memories]
            })
        
        # 5. Format context for Claude Code
        formatted_context = self._format_context_for_claude(context_parts, max_tokens)
        
        return {
            "formatted_context": formatted_context,
            "context_parts": context_parts,
            "total_memories": sum(len(part["memories"]) for part in context_parts),
            "session_id": current_session_id,
            "previous_session_id": previous_session_id
        }
    
    async def create_handoff_note(
        self,
        session_id: str,
        content: str,
        next_tasks: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None
    ) -> MemoryResponse:
        """
        Create a handoff note for the next Claude Code session
        
        Args:
            session_id: Current session ID
            content: Main handoff message
            next_tasks: List of tasks for next session
            warnings: Any warnings or important notes
        """
        handoff_data = {
            "handoff_type": "manual",
            "next_tasks": next_tasks or [],
            "warnings": warnings or [],
            "created_at": datetime.utcnow().isoformat()
        }
        
        formatted_content = f"HANDOFF NOTE: {content}"
        if next_tasks:
            formatted_content += f"\n\nNext Tasks:\n" + "\n".join(f"- {task}" for task in next_tasks)
        if warnings:
            formatted_content += f"\n\nWarnings:\n" + "\n".join(f"⚠️ {warning}" for warning in warnings)
        
        return await self.memory_service.create_memory(
            session_id=session_id,
            content=formatted_content,
            memory_type="decision",
            importance=0.95,
            metadata=handoff_data
        )
    
    async def get_session_chain(self, session_id: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        """Get the chain of linked sessions"""
        chain = []
        current_id = session_id
        depth = 0
        
        while current_id and depth < max_depth:
            session = await self.memory_service.get_session(current_id)
            if not session:
                break
                
            chain.append({
                "session_id": session["id"],
                "started_at": session.get("created_at"),
                "memory_count": session.get("memory_count", 0),
                "metadata": session.get("metadata", {})
            })
            
            # Get previous session ID from metadata
            current_id = session.get("metadata", {}).get("previous_session_id")
            depth += 1
            
        return chain
    
    def _create_handoff_summary(self, previous_memories: List[Memory]) -> str:
        """Create a summary of the previous session for handoff"""
        if not previous_memories:
            return "No previous context available."
        
        # Group by type
        by_type = {}
        for memory in previous_memories:
            mem_type = memory.memory_type
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory)
        
        # Create summary
        summary_parts = []
        
        # Decisions
        if "decision" in by_type:
            summary_parts.append(f"Key decisions: {len(by_type['decision'])} made")
        
        # Errors
        if "error" in by_type:
            summary_parts.append(f"Errors encountered: {len(by_type['error'])}")
        
        # Code changes
        if "code" in by_type:
            summary_parts.append(f"Code modifications: {len(by_type['code'])}")
        
        # Recent focus
        recent = sorted(previous_memories, key=lambda m: m.created_at, reverse=True)[:3]
        if recent:
            focus = recent[0].content[:100]
            summary_parts.append(f"Last focus: {focus}...")
        
        return " | ".join(summary_parts)
    
    def _memory_to_dict(self, memory: Memory) -> Dict[str, Any]:
        """Convert memory object to dictionary"""
        return {
            "id": str(memory.id),
            "content": memory.content,
            "type": memory.memory_type,
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "metadata": memory.metadata or {}
        }
    
    def _format_context_for_claude(self, context_parts: List[Dict], max_tokens: int) -> str:
        """Format context parts into a string for Claude Code"""
        formatted = "=== SESSION CONTEXT ===\n\n"
        
        for part in context_parts:
            if part["type"] == "handoff_notes":
                formatted += "📝 HANDOFF NOTES:\n"
                for mem in part["memories"]:
                    formatted += f"- {mem['content']}\n"
                formatted += "\n"
            
            elif part["type"] == "unfinished_tasks":
                formatted += "📋 UNFINISHED TASKS:\n"
                for mem in part["memories"]:
                    formatted += f"- {mem['content'][:150]}...\n"
                formatted += "\n"
            
            elif part["type"] == "important_context":
                formatted += "⚡ IMPORTANT CONTEXT:\n"
                for mem in part["memories"]:
                    formatted += f"- [{mem['type']}] {mem['content'][:100]}...\n"
                formatted += "\n"
            
            elif part["type"] == "recent_errors":
                formatted += "❌ RECENT ERRORS & SOLUTIONS:\n"
                for mem in part["memories"]:
                    formatted += f"- {mem['content'][:150]}...\n"
                formatted += "\n"
        
        # Truncate if needed
        if len(formatted) > max_tokens:
            formatted = formatted[:max_tokens-50] + "\n... [context truncated]"
        
        return formatted