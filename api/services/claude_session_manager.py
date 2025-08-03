"""
Claude Session Manager - Automatic session continuity and context restoration
"""

import os
import json
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc

from ..models.memory import MemoryItem
from .project_context_manager import ProjectContextManager
from .mistake_learning_system import MistakeLearningSystem
from ..path_config import MEMORY_CLI_PATH


class ClaudeSessionManager:
    """Manages Claude Code session continuity with automatic context restoration"""
    
    def __init__(self):
        self.memory_cli_path = MEMORY_CLI_PATH
        self.session_file = Path.home() / ".claude_session.json"
        self.context_file = Path.home() / ".claude_context.json"
        self.project_manager = ProjectContextManager()
        self.mistake_learner = MistakeLearningSystem()
        
    def _run_memory_cli(self, *args) -> Optional[str]:
        """Run memory-cli command and return output"""
        try:
            # Check if memory-cli exists and is executable
            if not os.path.exists(self.memory_cli_path):
                return None
            
            cmd = [self.memory_cli_path] + list(args)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # Silently fail - memory-cli is optional
            return None
    
    def _get_project_id(self, cwd: str) -> str:
        """Generate consistent project ID from path"""
        return hashlib.md5(str(Path(cwd).resolve()).encode()).hexdigest()[:12]
    
    def start_session(self, cwd: str, db: Optional[Session] = None) -> Dict[str, Any]:
        """
        Start a new Claude Code session with automatic context restoration
        """
        session_id = f"claude-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        project_id = self._get_project_id(cwd)
        project_path = Path(cwd)
        
        # Detect project type
        project_type = "unknown"
        if (project_path / "package.json").exists():
            project_type = "nodejs"
        elif (project_path / "requirements.txt").exists() or (project_path / "setup.py").exists():
            project_type = "python"
        elif (project_path / "Cargo.toml").exists():
            project_type = "rust"
        elif (project_path / "go.mod").exists():
            project_type = "go"
        
        # Load previous session info
        previous_session = None
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    previous_session = json.load(f)
            except:
                pass
        
        # Prepare session data
        session_data = {
            "session_id": session_id,
            "project_id": project_id,
            "project_path": str(project_path),
            "project_name": project_path.name,
            "project_type": project_type,
            "started_at": datetime.utcnow().isoformat(),
            "previous_session": previous_session.get("session_id") if previous_session else None
        }
        
        # Save current session
        with open(self.session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Switch to project context if db is provided
        project_context = None
        if db:
            try:
                project_context = self.project_manager.switch_project_context(str(project_path), db)
                # Update session data with project info
                if project_context and "project" in project_context:
                    session_data["project_type"] = project_context["project"].get("primary_language", project_type)
                    session_data["project_frameworks"] = project_context["project"].get("frameworks", [])
                    session_data["project_patterns"] = project_context["project"].get("patterns", {})
            except Exception as e:
                print(f"Project context switch failed: {e}")
        
        # Get context from memory system
        context = self._restore_context(project_id, previous_session)
        
        # Merge project context if available
        if project_context:
            context["project_conventions"] = project_context.get("preferences", {})
            context["project_patterns"] = project_context.get("project", {}).get("patterns", {})
            context["project_memories"] = [m["content"][:100] + "..." for m in project_context.get("memories", [])[:5]]
        
        # Store session start in memory (if memory-cli is available)
        if os.path.exists(self.memory_cli_path):
            self._run_memory_cli(
                "add",
                f"Claude Code session started: {session_id} for project {project_path.name}",
                "-t", "fact",
                "-p", "high"
            )
        
        return {
            "session": session_data,
            "context": context,
            "status": "Session initialized with restored context"
        }
    
    def _restore_context(self, project_id: str, previous_session: Optional[Dict]) -> Dict[str, Any]:
        """Restore context from memory system"""
        context = {
            "memories": [],
            "handoff_notes": [],
            "recent_errors": [],
            "unfinished_tasks": [],
            "project_patterns": []
        }
        
        # Get recent memories from memory CLI if available
        if os.path.exists(self.memory_cli_path):
            recent_output = self._run_memory_cli("context", "--limit", "20")
            if recent_output:
                context["memories"] = recent_output.split('\n')[:5]  # First 5 lines
            
            # Search for project-specific memories
            project_search = self._run_memory_cli("search", f"project {project_id}")
            if project_search:
                context["project_patterns"] = project_search.split('\n')[:3]
            
            # Look for handoff notes
            handoff_search = self._run_memory_cli("search", "handoff")
            if handoff_search:
                context["handoff_notes"] = handoff_search.split('\n')[:3]
            
            # Find recent errors
            error_search = self._run_memory_cli("search", "-t", "error")
            if error_search:
                context["recent_errors"] = error_search.split('\n')[:3]
            
            # Check for unfinished tasks from previous session
            if previous_session:
                task_search = self._run_memory_cli("search", f"TODO session:{previous_session.get('session_id')}")
                if task_search:
                    context["unfinished_tasks"] = task_search.split('\n')[:5]
        
        # Save context for quick access
        with open(self.context_file, 'w') as f:
            json.dump(context, f, indent=2)
        
        return context
    
    def create_handoff_note(self, content: str, next_tasks: List[str], 
                           unresolved_issues: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create comprehensive handoff note for next session"""
        session_data = {}
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
        
        session_id = session_data.get("session_id", "unknown")
        project_name = session_data.get("project_name", "unknown")
        
        # Build handoff content
        handoff_lines = [
            f"HANDOFF NOTE - Session {session_id}",
            f"Project: {project_name}",
            f"Date: {datetime.utcnow().isoformat()}",
            "",
            "Summary:",
            content,
            ""
        ]
        
        if next_tasks:
            handoff_lines.extend([
                "Next Tasks:",
                *[f"- TODO: {task}" for task in next_tasks],
                ""
            ])
        
        if unresolved_issues:
            handoff_lines.extend([
                "Unresolved Issues:",
                *[f"- ISSUE: {issue}" for issue in unresolved_issues],
                ""
            ])
        
        handoff_content = "\n".join(handoff_lines)
        
        # Store in memory system (if available)
        if os.path.exists(self.memory_cli_path):
            self._run_memory_cli(
                "add",
                handoff_content,
                "-t", "context",
                "-p", "critical"
            )
            
            # Create checkpoint
            self._run_memory_cli(
                "checkpoint",
                "-d", f"Session {session_id} handoff: {content[:50]}..."
            )
        
        return {
            "handoff_id": f"handoff-{session_id}",
            "content": handoff_content,
            "stored": True
        }
    
    def record_error_with_solution(self, error_type: str, error_message: str, 
                                  solution: Optional[str], worked: bool,
                                  db: Optional[Session] = None) -> Dict[str, Any]:
        """Record error and solution in memory system"""
        session_data = {}
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
        
        session_id = session_data.get("session_id", "unknown")
        project_name = session_data.get("project_name", "unknown")
        
        # Build error record
        error_content = [
            f"ERROR in {project_name} (session {session_id})",
            f"Type: {error_type}",
            f"Message: {error_message}"
        ]
        
        if solution:
            status = "WORKED" if worked else "FAILED"
            error_content.append(f"Solution [{status}]: {solution}")
        else:
            error_content.append("Solution: Not found yet")
        
        # Store in memory (if available)
        if os.path.exists(self.memory_cli_path):
            self._run_memory_cli(
                "add",
                "\n".join(error_content),
                "-t", "error",
                "-p", "high" if not worked else "medium"
            )
        
        # Track with mistake learning system if db available
        mistake_result = None
        if db:
            try:
                context = {
                    "session_id": session_id,
                    "project_name": project_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                mistake_result = self.mistake_learner.track_mistake(
                    db, error_type, error_message, context,
                    attempted_solution=None if worked else solution,
                    successful_solution=solution if worked else None,
                    project_id=session_data.get("project_id")
                )
            except Exception as e:
                print(f"Mistake tracking failed: {e}")
        
        return {
            "recorded": True,
            "error_id": f"error-{session_id}-{datetime.utcnow().timestamp()}",
            "mistake_tracked": mistake_result is not None,
            "is_repeated": mistake_result.get("is_repeated", False) if mistake_result else False,
            "repetition_count": mistake_result.get("repetition_count", 0) if mistake_result else 0
        }
    
    def get_similar_errors(self, error_type: str, error_message: str) -> List[Dict[str, Any]]:
        """Find similar errors with solutions from memory"""
        similar_errors = []
        
        # Search for similar errors (if memory-cli is available)
        if os.path.exists(self.memory_cli_path):
            search_results = self._run_memory_cli("search", f"ERROR Type: {error_type}")
            
            if search_results:
                lines = search_results.split('\n')
                current_error = {}
                
                for line in lines:
                    if line.startswith("ERROR"):
                        if current_error:
                            similar_errors.append(current_error)
                        current_error = {"description": line}
                    elif "Solution [WORKED]:" in line:
                        current_error["solution"] = line.split("Solution [WORKED]:")[1].strip()
                        current_error["worked"] = True
                    elif "Solution [FAILED]:" in line:
                        current_error["solution"] = line.split("Solution [FAILED]:")[1].strip()
                        current_error["worked"] = False
                
                if current_error:
                    similar_errors.append(current_error)
        
        return similar_errors[:5]  # Return top 5
    
    def predict_next_tasks(self) -> List[Dict[str, Any]]:
        """Predict next tasks based on context"""
        predictions = []
        
        # Load current context
        if self.context_file.exists():
            with open(self.context_file, 'r') as f:
                context = json.load(f)
            
            # Add unfinished tasks
            for task in context.get("unfinished_tasks", []):
                if "TODO:" in task:
                    predictions.append({
                        "task": task.split("TODO:")[1].strip(),
                        "type": "unfinished",
                        "confidence": 0.9
                    })
            
            # Add tasks from handoff notes
            for note in context.get("handoff_notes", []):
                if "TODO:" in note:
                    predictions.append({
                        "task": note.split("TODO:")[1].strip(),
                        "type": "handoff",
                        "confidence": 0.8
                    })
            
            # Add error fixes
            for error in context.get("recent_errors", []):
                if "Solution: Not found yet" in error:
                    error_type = error.split('Type:')[1].split('\n')[0].strip() if 'Type:' in error else "Unknown"
                    predictions.append({
                        "task": f"Fix error: {error_type}",
                        "type": "error_fix",
                        "confidence": 0.7
                    })
        
        return predictions[:5]
    
    def end_session(self, summary: Optional[str] = None) -> Dict[str, Any]:
        """End current session and prepare for next"""
        session_data = {}
        if self.session_file.exists():
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
        
        session_id = session_data.get("session_id", "unknown")
        
        # Create session summary (if memory-cli is available)
        if summary and os.path.exists(self.memory_cli_path):
            self._run_memory_cli(
                "add",
                f"Session {session_id} ended: {summary}",
                "-t", "summary",
                "-p", "high"
            )
        
        # Get session stats (if available)
        stats_output = None
        if os.path.exists(self.memory_cli_path):
            stats_output = self._run_memory_cli("stats")
        
        return {
            "session_id": session_id,
            "ended_at": datetime.utcnow().isoformat(),
            "summary_stored": bool(summary),
            "memory_stats": stats_output
        }
    
    def get_project_context(self, cwd: str, db: Session) -> Dict[str, Any]:
        """Get comprehensive project context from both memory systems"""
        project_id = self._get_project_id(cwd)
        project_path = Path(cwd)
        
        # Get from local memory system
        local_context = self._restore_context(project_id, None)
        
        # Get from KnowledgeHub if available
        kb_memories = []
        try:
            memories = db.query(MemoryItem).filter(
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            ).order_by(desc(MemoryItem.created_at)).limit(10).all()
            
            kb_memories = [{
                "content": m.content[:200],
                "type": m.meta_data.get("memory_type", "unknown"),
                "created": m.created_at.isoformat()
            } for m in memories]
        except Exception as e:
            print(f"Failed to get KnowledgeHub memories: {e}")
        
        return {
            "project": {
                "id": project_id,
                "path": str(project_path),
                "name": project_path.name
            },
            "local_memory": local_context,
            "knowledgehub_memories": kb_memories,
            "combined_count": len(local_context.get("memories", [])) + len(kb_memories)
        }