"""
Project Context Manager - Per-project memory isolation and context switching
"""

import os
import json
import hashlib
import subprocess
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import cast, String, desc, and_

from ..models.memory import MemoryItem
from ..path_config import MEMORY_CLI_PATH


class ProjectContextManager:
    """Manages project-specific contexts with memory isolation"""
    
    def __init__(self):
        self.projects_dir = Path.home() / ".claude_projects"
        self.projects_dir.mkdir(exist_ok=True)
        self.active_project_file = Path.home() / ".claude_active_project.json"
        self.memory_cli_path = MEMORY_CLI_PATH
        
    def _get_project_id(self, project_path: str) -> str:
        """Generate consistent project ID from path"""
        abs_path = str(Path(project_path).resolve())
        return hashlib.md5(abs_path.encode()).hexdigest()[:12]
    
    def _get_project_config_path(self, project_id: str) -> Path:
        """Get path to project config file"""
        return self.projects_dir / f"{project_id}.json"
    
    def _detect_project_type(self, project_path: Path) -> Dict[str, Any]:
        """Detect project type and characteristics"""
        indicators = {
            "python": ["requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
            "nodejs": ["package.json", "yarn.lock", "package-lock.json"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "go": ["go.mod", "go.sum"],
            "java": ["pom.xml", "build.gradle", "build.gradle.kts"],
            "csharp": ["*.csproj", "*.sln"],
            "ruby": ["Gemfile", "Gemfile.lock"],
            "php": ["composer.json", "composer.lock"]
        }
        
        detected_types = []
        for lang, files in indicators.items():
            for file_pattern in files:
                if '*' in file_pattern:
                    if list(project_path.glob(file_pattern)):
                        detected_types.append(lang)
                        break
                elif (project_path / file_pattern).exists():
                    detected_types.append(lang)
                    break
        
        # Detect frameworks
        frameworks = []
        if "python" in detected_types:
            if (project_path / "manage.py").exists():
                frameworks.append("django")
            elif (project_path / "app.py").exists() or (project_path / "application.py").exists():
                frameworks.append("flask")
            elif (project_path / "main.py").exists() and (project_path / "requirements.txt").exists():
                with open(project_path / "requirements.txt") as f:
                    reqs = f.read()
                    if "fastapi" in reqs:
                        frameworks.append("fastapi")
        
        if "nodejs" in detected_types and (project_path / "package.json").exists():
            with open(project_path / "package.json") as f:
                pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                if "react" in deps:
                    frameworks.append("react")
                if "vue" in deps:
                    frameworks.append("vue")
                if "express" in deps:
                    frameworks.append("express")
                if "@angular/core" in deps:
                    frameworks.append("angular")
        
        # Detect test frameworks
        test_frameworks = []
        if "python" in detected_types:
            if (project_path / "pytest.ini").exists() or (project_path / "conftest.py").exists():
                test_frameworks.append("pytest")
            elif (project_path / "tests").exists():
                test_frameworks.append("unittest")
        
        return {
            "languages": detected_types,
            "primary_language": detected_types[0] if detected_types else "unknown",
            "frameworks": frameworks,
            "test_frameworks": test_frameworks,
            "has_docker": (project_path / "Dockerfile").exists() or (project_path / "docker-compose.yml").exists(),
            "has_ci": any((project_path / ci).exists() for ci in [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile"]),
            "detected_at": datetime.utcnow().isoformat()
        }
    
    def create_project_profile(self, project_path: str, db: Session) -> Dict[str, Any]:
        """Create or update project profile with isolation"""
        project_path = Path(project_path).resolve()
        project_id = self._get_project_id(str(project_path))
        config_path = self._get_project_config_path(project_id)
        
        # Load existing config or create new
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                "id": project_id,
                "path": str(project_path),
                "name": project_path.name,
                "created_at": datetime.utcnow().isoformat(),
                "sessions": [],
                "patterns": {},
                "preferences": {},
                "common_errors": {},
                "conventions": {},
                "dependencies": set(),
                "memory_namespace": f"project_{project_id}"
            }
        
        # Update with current detection
        detection = self._detect_project_type(project_path)
        config.update(detection)
        config["last_accessed"] = datetime.utcnow().isoformat()
        
        # Detect patterns from existing code
        patterns = self._detect_code_patterns(project_path, config.get("primary_language", "unknown"))
        config["patterns"].update(patterns)
        
        # Save config
        # Convert sets to lists for JSON serialization
        save_config = config.copy()
        if isinstance(save_config.get("dependencies"), set):
            save_config["dependencies"] = list(save_config["dependencies"])
        
        with open(config_path, 'w') as f:
            json.dump(save_config, f, indent=2)
        
        # Store in memory with project namespace
        self._store_project_memory(
            db,
            project_id,
            f"Project Profile Updated: {config['name']}",
            {
                "profile_type": "project_config",
                "languages": config.get("languages", []),
                "frameworks": config.get("frameworks", []),
                "patterns": patterns
            }
        )
        
        return config
    
    def _detect_code_patterns(self, project_path: Path, language: str) -> Dict[str, Any]:
        """Detect coding patterns and conventions"""
        patterns = {
            "indent_style": "unknown",
            "quote_style": "unknown",
            "naming_convention": "unknown",
            "test_pattern": "unknown"
        }
        
        if language == "python":
            # Check for indent style
            py_files = list(project_path.glob("**/*.py"))[:10]  # Sample first 10
            space_count = 0
            tab_count = 0
            
            for py_file in py_files:
                try:
                    with open(py_file, 'r') as f:
                        for line in f:
                            if line.startswith('    '):
                                space_count += 1
                            elif line.startswith('\t'):
                                tab_count += 1
                except:
                    continue
            
            patterns["indent_style"] = "spaces" if space_count > tab_count else "tabs"
            
            # Check naming convention
            if any(f.name.startswith("test_") for f in py_files):
                patterns["test_pattern"] = "test_prefix"
            elif any(f.name.endswith("_test.py") for f in py_files):
                patterns["test_pattern"] = "test_suffix"
        
        elif language == "nodejs":
            # Check quote style from package.json or JS files
            js_files = list(project_path.glob("**/*.js"))[:10]
            single_quotes = 0
            double_quotes = 0
            
            for js_file in js_files:
                try:
                    with open(js_file, 'r') as f:
                        content = f.read()
                        single_quotes += content.count("'")
                        double_quotes += content.count('"')
                except:
                    continue
            
            patterns["quote_style"] = "single" if single_quotes > double_quotes else "double"
        
        return patterns
    
    def switch_project_context(self, project_path: str, db: Session) -> Dict[str, Any]:
        """Switch to a different project context"""
        project_id = self._get_project_id(project_path)
        
        # Save current project state if exists
        if self.active_project_file.exists():
            with open(self.active_project_file, 'r') as f:
                current = json.load(f)
            if current.get("id") != project_id:
                self._save_project_state(current["id"], db)
        
        # Load or create project profile
        profile = self.create_project_profile(project_path, db)
        
        # Set as active project
        with open(self.active_project_file, 'w') as f:
            json.dump({
                "id": project_id,
                "path": project_path,
                "activated_at": datetime.utcnow().isoformat()
            }, f)
        
        # Load project-specific memories
        memories = self._load_project_memories(project_id, db)
        
        # Get project-specific preferences
        preferences = profile.get("preferences", {})
        
        return {
            "project": profile,
            "memories": memories,
            "preferences": preferences,
            "context_switched": True
        }
    
    def _store_project_memory(self, db: Session, project_id: str, content: str, 
                             metadata: Dict[str, Any]) -> MemoryItem:
        """Store memory with project isolation"""
        # Add project namespace to metadata
        metadata["project_id"] = project_id
        metadata["namespace"] = f"project_{project_id}"
        
        memory_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Check if exists in this project's namespace
        existing = db.query(MemoryItem).filter(
            and_(
                MemoryItem.content_hash == memory_hash,
                cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
            )
        ).first()
        
        if existing:
            existing.access_count += 1
            existing.accessed_at = datetime.utcnow()
            db.commit()
            return existing
        
        # Create new memory
        memory = MemoryItem(
            content=content,
            content_hash=memory_hash,
            tags=[f"project:{project_id}"],
            meta_data=metadata,
            access_count=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            accessed_at=datetime.utcnow()
        )
        db.add(memory)
        db.commit()
        db.refresh(memory)
        
        return memory
    
    def _load_project_memories(self, project_id: str, db: Session, limit: int = 50) -> List[Dict[str, Any]]:
        """Load memories specific to a project"""
        memories = db.query(MemoryItem).filter(
            cast(MemoryItem.meta_data, String).contains(f'"project_id": "{project_id}"')
        ).order_by(desc(MemoryItem.accessed_at)).limit(limit).all()
        
        return [{
            "id": str(m.id),
            "content": m.content,
            "type": m.meta_data.get("memory_type", "unknown"),
            "created": m.created_at.isoformat(),
            "accessed": m.accessed_at.isoformat() if m.accessed_at else None,
            "tags": m.tags
        } for m in memories]
    
    def _save_project_state(self, project_id: str, db: Session):
        """Save current project state before switching"""
        # This could save open files, cursor positions, etc.
        # For now, just update last accessed time
        config_path = self._get_project_config_path(project_id)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            config["last_accessed"] = datetime.utcnow().isoformat()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def add_project_preference(self, project_path: str, key: str, value: Any) -> Dict[str, Any]:
        """Add a project-specific preference"""
        project_id = self._get_project_id(project_path)
        config_path = self._get_project_config_path(project_id)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"preferences": {}}
        
        config["preferences"][key] = value
        config["preferences_updated"] = datetime.utcnow().isoformat()
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config["preferences"]
    
    def record_project_pattern(self, project_path: str, pattern_type: str, 
                              pattern_value: str, db: Session) -> Dict[str, Any]:
        """Record a discovered pattern for the project"""
        project_id = self._get_project_id(project_path)
        
        # Store in project config
        config_path = self._get_project_config_path(project_id)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"patterns": {}}
        
        if "patterns" not in config:
            config["patterns"] = {}
        
        config["patterns"][pattern_type] = pattern_value
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Also store in memory for learning
        self._store_project_memory(
            db,
            project_id,
            f"Pattern discovered: {pattern_type} = {pattern_value}",
            {
                "memory_type": "pattern",
                "pattern_type": pattern_type,
                "pattern_value": pattern_value,
                "importance": 0.8
            }
        )
        
        return config["patterns"]
    
    def get_project_conventions(self, project_path: str) -> Dict[str, Any]:
        """Get all conventions and patterns for a project"""
        project_id = self._get_project_id(project_path)
        config_path = self._get_project_config_path(project_id)
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return {
            "patterns": config.get("patterns", {}),
            "preferences": config.get("preferences", {}),
            "conventions": config.get("conventions", {}),
            "languages": config.get("languages", []),
            "frameworks": config.get("frameworks", [])
        }
    
    def list_all_projects(self) -> List[Dict[str, Any]]:
        """List all known projects"""
        projects = []
        
        for config_file in self.projects_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                projects.append({
                    "id": config.get("id"),
                    "name": config.get("name"),
                    "path": config.get("path"),
                    "language": config.get("primary_language"),
                    "last_accessed": config.get("last_accessed"),
                    "sessions": len(config.get("sessions", []))
                })
            except:
                continue
        
        # Sort by last accessed
        projects.sort(key=lambda x: x.get("last_accessed", ""), reverse=True)
        
        return projects
    
    def get_active_project(self) -> Optional[Dict[str, Any]]:
        """Get currently active project"""
        if not self.active_project_file.exists():
            return None
        
        try:
            with open(self.active_project_file, 'r') as f:
                return json.load(f)
        except:
            return None