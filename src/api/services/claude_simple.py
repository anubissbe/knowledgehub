"""
Simplified Claude Code Enhancement Services
Working implementation that integrates with existing memory system
"""

import os
import json
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, cast, String

from ..models.memory import MemoryItem
from .claude_memory_adapter import ClaudeMemoryAdapter


class ClaudeEnhancementService:
    """Simplified service providing all Claude enhancements"""
    
    def __init__(self, db: Session):
        self.db = db
        self.adapter = ClaudeMemoryAdapter(db)
    
    # ========== Session Continuity ==========
    
    async def continue_session(self, previous_session_id: str) -> Dict[str, Any]:
        """Continue from a previous session"""
        # Create new session ID
        new_session_id = f"session-{datetime.utcnow().isoformat()}"
        
        # Get previous session memories
        previous_memories = self.adapter.get_memories_by_session(previous_session_id, limit=20)
        
        # Create continuation memory
        handoff_content = f"SESSION CONTINUATION: Continuing from {previous_session_id}"
        if previous_memories:
            last_memory = previous_memories[0]
            handoff_content += f"\nLast activity: {last_memory.content[:100]}..."
        
        await self.adapter.create_memory(
            content=handoff_content,
            memory_type="decision",
            importance=0.9,
            session_id=new_session_id,
            metadata={"previous_session": previous_session_id}
        )
        
        # Get relevant context
        context = self._format_session_context(previous_memories)
        
        return {
            "session_id": new_session_id,
            "previous_session_id": previous_session_id,
            "context": context,
            "memory_count": len(previous_memories)
        }
    
    async def create_handoff_note(
        self, session_id: str, content: str, 
        next_tasks: Optional[List[str]] = None
    ) -> MemoryItem:
        """Create a handoff note for next session"""
        handoff_content = f"HANDOFF NOTE: {content}"
        if next_tasks:
            handoff_content += "\nNext tasks:\n" + "\n".join(f"- {task}" for task in next_tasks)
        
        return await self.adapter.create_memory(
            content=handoff_content,
            memory_type="decision",
            importance=0.95,
            session_id=session_id,
            tags=["handoff", "session"],
            metadata={"handoff": True, "next_tasks": next_tasks}
        )
    
    # ========== Project Profiles ==========
    
    async def detect_project(self, cwd: str) -> Dict[str, Any]:
        """Detect and profile current project"""
        project_path = Path(cwd)
        project_id = hashlib.md5(str(project_path).encode()).hexdigest()[:12]
        
        # Detect project type
        project_info = {
            "id": project_id,
            "path": str(project_path),
            "name": project_path.name,
            "type": "unknown",
            "language": "unknown"
        }
        
        # Check for common project files
        if (project_path / "package.json").exists():
            project_info["type"] = "node"
            project_info["language"] = "javascript"
        elif (project_path / "requirements.txt").exists() or (project_path / "setup.py").exists():
            project_info["type"] = "python"
            project_info["language"] = "python"
        elif (project_path / "Cargo.toml").exists():
            project_info["type"] = "rust"
            project_info["language"] = "rust"
        
        # Store project profile
        await self.adapter.create_memory(
            content=f"Project Profile: {project_info['name']} ({project_info['type']})",
            memory_type="entity",
            importance=0.8,
            project_id=project_id,
            tags=["project", project_info["type"]],
            metadata=project_info
        )
        
        return project_info
    
    async def get_project_context(self, project_id: str) -> Dict[str, Any]:
        """Get all context for a project"""
        # Use adapter method which has correct JSONB query
        memories = self.adapter.get_memories_by_project(project_id, limit=50)
        
        # Group by type
        by_type = {}
        for mem in memories:
            mem_type = mem.meta_data.get("memory_type", "unknown")
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(mem.content[:100])
        
        return {
            "project_id": project_id,
            "total_memories": len(memories),
            "by_type": by_type,
            "recent": [m.content[:100] for m in memories[:5]]
        }
    
    # ========== Error Learning ==========
    
    async def record_error(
        self, error_type: str, error_message: str,
        solution: Optional[str] = None, success: bool = False,
        session_id: Optional[str] = None, project_id: Optional[str] = None
    ) -> MemoryItem:
        """Record an error and its solution"""
        content = f"ERROR [{error_type}]: {error_message}"
        if solution:
            content += f"\nSOLUTION: {solution} ({'✓' if success else '✗'})"
        
        return await self.adapter.create_memory(
            content=content,
            memory_type="error",
            importance=0.8 if not success else 0.9,
            session_id=session_id,
            project_id=project_id,
            tags=["error", error_type.lower()],
            metadata={
                "error_type": error_type,
                "solution": solution,
                "success": success
            }
        )
    
    async def find_similar_errors(self, error_type: str, error_message: str) -> List[Dict[str, Any]]:
        """Find similar errors and their solutions"""
        # Search for similar errors
        error_memories = self.adapter.search_memories(error_type, limit=10)
        
        similar = []
        for mem in error_memories:
            if mem.meta_data.get("memory_type") == "error":
                similar.append({
                    "id": str(mem.id),
                    "error": mem.content[:200],
                    "solution": mem.meta_data.get("solution"),
                    "success": mem.meta_data.get("success", False),
                    "created": mem.created_at.isoformat()
                })
        
        return similar
    
    # ========== Task Prediction ==========
    
    async def predict_next_tasks(self, session_id: str, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Predict likely next tasks"""
        predictions = []
        
        # Check for unfinished tasks (handoff notes)
        from sqlalchemy import cast, String
        handoff_memories = self.db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains('"handoff": true')
        ).order_by(desc(MemoryItem.created_at)).limit(5).all()
        
        for mem in handoff_memories:
            next_tasks = mem.meta_data.get("next_tasks", [])
            for task in next_tasks:
                predictions.append({
                    "task": task,
                    "type": "handoff",
                    "confidence": 0.9,
                    "from_session": mem.meta_data.get("session_id")
                })
        
        # Check recent errors without solutions
        unsolved_errors = self.db.query(MemoryItem).filter(
            and_(
                cast(MemoryItem.meta_data, String).contains('"memory_type": "error"'),
                cast(MemoryItem.meta_data, String).contains('"success": false')
            )
        ).order_by(desc(MemoryItem.created_at)).limit(3).all()
        
        for error in unsolved_errors:
            predictions.append({
                "task": f"Fix error: {error.meta_data.get('error_type', 'Unknown')}",
                "type": "error_fix",
                "confidence": 0.7,
                "error_id": str(error.id)
            })
        
        return predictions[:5]  # Top 5 predictions
    
    # ========== Unified Initialize ==========
    
    async def initialize_claude(self, cwd: str, previous_session_id: Optional[str] = None) -> Dict[str, Any]:
        """Initialize Claude with all enhancements"""
        result = {
            "initialized_at": datetime.utcnow().isoformat()
        }
        
        # Detect project
        project = await self.detect_project(cwd)
        result["project"] = project
        
        # Continue or start session
        if previous_session_id:
            session_data = await self.continue_session(previous_session_id)
        else:
            session_data = {
                "session_id": f"session-{datetime.utcnow().isoformat()}",
                "context": "Starting fresh session"
            }
        result["session"] = session_data
        
        # Get project context
        project_context = await self.get_project_context(project["id"])
        result["project_context"] = project_context
        
        # Predict tasks
        predictions = await self.predict_next_tasks(
            session_data["session_id"],
            project["id"]
        )
        result["predicted_tasks"] = predictions
        
        return result
    
    # ========== Helper Methods ==========
    
    def _format_session_context(self, memories: List[MemoryItem]) -> str:
        """Format memories into context string"""
        if not memories:
            return "No previous context available."
        
        context_parts = ["=== SESSION CONTEXT ===\n"]
        
        # Group by type
        by_type = {}
        for mem in memories:
            mem_type = mem.meta_data.get("memory_type", "unknown")
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(mem)
        
        # Format each type
        for mem_type, mems in by_type.items():
            context_parts.append(f"\n{mem_type.upper()} ({len(mems)} items):")
            for mem in mems[:3]:  # Top 3 per type
                context_parts.append(f"- {mem.content[:100]}...")
        
        return "\n".join(context_parts)