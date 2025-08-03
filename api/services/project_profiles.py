"""
Project Profiles Service for Claude Code
Manages project-specific context and preferences
"""

import os
import json
import hashlib
from typing import Optio, TYPE_CHECKINGnal, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from ..models.memory import Memory
from .memory_service import MemoryService


class ProjectProfilesService:
    """Manages project-specific context profiles for Claude Code"""
    
    def __init__(self, db: Session, memory_service: MemoryService):
        self.db = db
        self.memory_service = memory_service
        
    async def detect_project(self, cwd: str) -> Dict[str, Any]:
        """
        Detect project from current working directory
        
        Args:
            cwd: Current working directory
            
        Returns:
            Project profile information
        """
        # Normalize path
        project_path = Path(cwd).resolve()
        
        # Look for project markers
        project_info = self._analyze_project_structure(project_path)
        
        # Generate project ID
        project_id = self._generate_project_id(str(project_path))
        
        # Get or create project profile
        profile = await self._get_or_create_profile(project_id, project_info)
        
        return profile
    
    async def load_context(self, project_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load all relevant context for a project
        
        Args:
            project_id: The project identifier
            session_id: Optional session to associate context with
            
        Returns:
            Project-specific context
        """
        context = {
            "project_id": project_id,
            "loaded_at": datetime.utcnow().isoformat(),
            "memories": {},
            "patterns": {},
            "preferences": {}
        }
        
        # 1. Load project configuration memories
        config_memories = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.memory_type == "preference"
            )
        ).order_by(desc(Memory.importance)).all()
        
        context["preferences"] = {
            mem.metadata.get("preference_key", f"pref_{mem.id}"): mem.content 
            for mem in config_memories
        }
        
        # 2. Load project-specific patterns
        pattern_memories = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.memory_type == "pattern"
            )
        ).order_by(desc(Memory.access_count)).limit(20).all()
        
        context["patterns"] = [
            {
                "pattern": mem.content,
                "usage_count": mem.access_count,
                "success_rate": mem.metadata.get("success_rate", 0.0)
            }
            for mem in pattern_memories
        ]
        
        # 3. Load recent project activities
        recent_memories = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.created_at > datetime.utcnow() - timedelta(days=30)
            )
        ).order_by(desc(Memory.created_at)).limit(50).all()
        
        # Group by type
        by_type = {}
        for mem in recent_memories:
            mem_type = mem.memory_type
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append({
                "content": mem.content,
                "importance": mem.importance,
                "created_at": mem.created_at.isoformat()
            })
        
        context["memories"] = by_type
        
        # 4. Load project-specific errors and solutions
        error_memories = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.memory_type == "error"
            )
        ).order_by(desc(Memory.created_at)).limit(20).all()
        
        context["known_errors"] = [
            {
                "error": mem.content,
                "solution": mem.metadata.get("solution", ""),
                "occurrences": mem.metadata.get("occurrences", 1)
            }
            for mem in error_memories
        ]
        
        # 5. Format for Claude Code
        formatted_context = self._format_project_context(context)
        
        # 6. Record context load if session provided
        if session_id:
            await self.memory_service.create_memory(
                session_id=session_id,
                content=f"Loaded project context for: {context.get('project_name', project_id)}",
                memory_type="fact",
                importance=0.3,
                metadata={
                    "project_id": project_id,
                    "context_load": True,
                    "memory_count": sum(len(mems) for mems in context["memories"].values())
                }
            )
        
        return {
            "project_id": project_id,
            "context": context,
            "formatted_context": formatted_context
        }
    
    async def save_project_preference(
        self,
        project_id: str,
        preference_key: str,
        preference_value: Any,
        session_id: Optional[str] = None
    ) -> Memory:
        """Save a project-specific preference"""
        content = f"Project preference: {preference_key} = {json.dumps(preference_value)}"
        
        # Check if preference exists
        existing = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.metadata.contains({"preference_key": preference_key}),
                Memory.memory_type == "preference"
            )
        ).first()
        
        if existing:
            # Update existing
            existing.content = content
            existing.metadata["preference_value"] = preference_value
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        
        # Create new
        return await self.memory_service.create_memory(
            session_id=session_id or "project-profile",
            content=content,
            memory_type="preference",
            importance=0.7,
            metadata={
                "project_id": project_id,
                "preference_key": preference_key,
                "preference_value": preference_value
            }
        )
    
    async def record_project_pattern(
        self,
        project_id: str,
        pattern: str,
        pattern_type: str,
        success: bool = True,
        session_id: Optional[str] = None
    ) -> Memory:
        """Record a project-specific pattern or approach"""
        # Check if pattern exists
        existing = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.content == pattern,
                Memory.memory_type == "pattern"
            )
        ).first()
        
        if existing:
            # Update statistics
            stats = existing.metadata.get("statistics", {"success": 0, "failure": 0})
            if success:
                stats["success"] += 1
            else:
                stats["failure"] += 1
            
            total = stats["success"] + stats["failure"]
            existing.metadata["statistics"] = stats
            existing.metadata["success_rate"] = stats["success"] / total if total > 0 else 0
            existing.access_count += 1
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing
        
        # Create new pattern
        return await self.memory_service.create_memory(
            session_id=session_id or "project-profile",
            content=pattern,
            memory_type="pattern",
            importance=0.6,
            metadata={
                "project_id": project_id,
                "pattern_type": pattern_type,
                "statistics": {"success": 1 if success else 0, "failure": 0 if success else 1},
                "success_rate": 1.0 if success else 0.0
            }
        )
    
    async def get_project_summary(self, project_id: str) -> Dict[str, Any]:
        """Get a summary of project knowledge"""
        summary = {
            "project_id": project_id,
            "statistics": {},
            "recent_focus": [],
            "top_patterns": [],
            "common_errors": []
        }
        
        # Get memory statistics
        memory_counts = self.db.query(
            Memory.memory_type,
            func.count(Memory.id).label('count')
        ).filter(
            Memory.metadata.contains({"project_id": project_id})
        ).group_by(Memory.memory_type).all()
        
        summary["statistics"] = {row[0]: row[1] for row in memory_counts}
        
        # Get recent focus areas
        recent = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.created_at > datetime.utcnow() - timedelta(days=7)
            )
        ).order_by(desc(Memory.created_at)).limit(10).all()
        
        summary["recent_focus"] = [
            {"content": mem.content[:100], "type": mem.memory_type}
            for mem in recent
        ]
        
        # Get top patterns
        patterns = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.memory_type == "pattern"
            )
        ).order_by(desc(Memory.access_count)).limit(5).all()
        
        summary["top_patterns"] = [
            {
                "pattern": pat.content,
                "usage": pat.access_count,
                "success_rate": pat.metadata.get("success_rate", 0.0)
            }
            for pat in patterns
        ]
        
        return summary
    
    def _analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project structure to determine type and characteristics"""
        info = {
            "path": str(project_path),
            "name": project_path.name,
            "type": "unknown",
            "language": "unknown",
            "framework": None,
            "markers": []
        }
        
        # Check for common project files
        markers = {
            "package.json": ("javascript", "node"),
            "requirements.txt": ("python", None),
            "setup.py": ("python", None),
            "pyproject.toml": ("python", None),
            "Cargo.toml": ("rust", None),
            "go.mod": ("go", None),
            "pom.xml": ("java", "maven"),
            "build.gradle": ("java", "gradle"),
            "composer.json": ("php", "composer"),
            "Gemfile": ("ruby", "bundler"),
            ".csproj": ("csharp", "dotnet")
        }
        
        for marker, (lang, framework) in markers.items():
            if (project_path / marker).exists():
                info["markers"].append(marker)
                info["language"] = lang
                if framework:
                    info["framework"] = framework
        
        # Check for specific frameworks
        if (project_path / "package.json").exists():
            try:
                with open(project_path / "package.json") as f:
                    pkg = json.load(f)
                    deps = pkg.get("dependencies", {})
                    dev_deps = pkg.get("devDependencies", {})
                    all_deps = {**deps, **dev_deps}
                    
                    if "react" in all_deps:
                        info["framework"] = "react"
                    elif "vue" in all_deps:
                        info["framework"] = "vue"
                    elif "express" in all_deps:
                        info["framework"] = "express"
                    elif "@angular/core" in all_deps:
                        info["framework"] = "angular"
            except:
                pass
        
        # Determine project type
        if ".git" in os.listdir(project_path):
            info["type"] = "git_repository"
        elif any(marker in info["markers"] for marker in ["package.json", "requirements.txt", "setup.py"]):
            info["type"] = "application"
        elif any(f.endswith((".md", ".txt", ".rst")) for f in os.listdir(project_path)):
            info["type"] = "documentation"
        
        return info
    
    def _generate_project_id(self, project_path: str) -> str:
        """Generate consistent project ID from path"""
        return hashlib.md5(project_path.encode()).hexdigest()[:12]
    
    async def _get_or_create_profile(self, project_id: str, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get existing profile or create new one"""
        # Check for existing profile
        profile_memory = self.db.query(Memory).filter(
            and_(
                Memory.metadata.contains({"project_id": project_id}),
                Memory.metadata.contains({"is_profile": True})
            )
        ).first()
        
        if profile_memory:
            return {
                "project_id": project_id,
                "exists": True,
                **profile_memory.metadata.get("project_info", project_info)
            }
        
        # Create new profile
        profile_content = f"Project Profile: {project_info['name']} ({project_info['type']})"
        if project_info['language'] != 'unknown':
            profile_content += f" - {project_info['language']}"
            if project_info['framework']:
                profile_content += f"/{project_info['framework']}"
        
        await self.memory_service.create_memory(
            session_id="project-profile",
            content=profile_content,
            memory_type="entity",
            importance=0.8,
            metadata={
                "project_id": project_id,
                "is_profile": True,
                "project_info": project_info,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        return {
            "project_id": project_id,
            "exists": False,
            **project_info
        }
    
    def _format_project_context(self, context: Dict[str, Any]) -> str:
        """Format project context for Claude Code"""
        formatted = f"=== PROJECT CONTEXT: {context.get('project_name', context['project_id'])} ===\n\n"
        
        # Preferences
        if context["preferences"]:
            formatted += "📋 PROJECT PREFERENCES:\n"
            for key, value in context["preferences"].items():
                formatted += f"  • {key}: {value}\n"
            formatted += "\n"
        
        # Common patterns
        if context["patterns"]:
            formatted += "🔄 COMMON PATTERNS:\n"
            for pattern in context["patterns"][:5]:
                success_rate = pattern.get("success_rate", 0) * 100
                formatted += f"  • {pattern['pattern'][:80]}... (used {pattern['usage_count']}x, {success_rate:.0f}% success)\n"
            formatted += "\n"
        
        # Known errors
        if context.get("known_errors"):
            formatted += "⚠️ KNOWN ERRORS:\n"
            for error in context["known_errors"][:5]:
                formatted += f"  • Error: {error['error'][:60]}...\n"
                if error.get("solution"):
                    formatted += f"    Solution: {error['solution'][:60]}...\n"
            formatted += "\n"
        
        # Recent activity summary
        if context["memories"]:
            formatted += "📊 RECENT ACTIVITY:\n"
            for mem_type, memories in context["memories"].items():
                if memories:
                    formatted += f"  • {mem_type}: {len(memories)} items\n"
        
        return formatted