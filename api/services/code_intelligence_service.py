#!/usr/bin/env python3
"""
Code Intelligence Service - Serena-inspired semantic code understanding for KnowledgeHub
Integrates LSP, semantic analysis, and project-aware memory for intelligent code operations
"""

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from pydantic import BaseModel
from pylsp import lsp
from pylsp.workspace import Document, Workspace
from pylsp.config import config as lsp_config

logger = logging.getLogger(__name__)


@dataclass
class CodeSymbol:
    """Represents a code symbol with semantic information"""
    name: str
    kind: str  # class, function, method, variable, etc.
    path: str  # file path
    line_start: int
    line_end: int
    name_path: str  # hierarchical path like "MyClass/my_method"
    docstring: Optional[str] = None
    signature: Optional[str] = None
    children: List['CodeSymbol'] = None
    references: List[Dict] = None
    
    def to_dict(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "name_path": self.name_path,
            "docstring": self.docstring,
            "signature": self.signature,
            "children": [c.to_dict() for c in self.children] if self.children else [],
            "references": self.references or []
        }


@dataclass
class ProjectContext:
    """Maintains project-specific context and memory"""
    project_root: str
    language: str
    framework: Optional[str] = None
    dependencies: List[str] = None
    symbols_cache: Dict[str, CodeSymbol] = None
    memory_items: Dict[str, str] = None
    last_analysis: Optional[datetime] = None
    
    def __post_init__(self):
        if self.symbols_cache is None:
            self.symbols_cache = {}
        if self.memory_items is None:
            self.memory_items = {}
        if self.dependencies is None:
            self.dependencies = []


class CodeIntelligenceService:
    """
    Serena-inspired code intelligence service for KnowledgeHub
    Provides semantic code understanding, LSP integration, and intelligent memory
    """
    
    def __init__(self, workspace_root: str = "/app"):
        self.workspace_root = Path(workspace_root)
        # Path mapping for containerized environment
        self.path_mappings = {
            "/opt/projects/knowledgehub": "/app/project",
            "/opt/projects": "/app"
        }
        self.projects: Dict[str, ProjectContext] = {}
        self.lsp_servers: Dict[str, subprocess.Popen] = {}
        self.workspaces: Dict[str, Workspace] = {}
        self.memory_dir = Path("/opt/claude/.serena_memories")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Language server configurations
        self.lsp_configs = {
            "python": {
                "cmd": ["pylsp"],
                "extensions": [".py"],
                "language_id": "python"
            },
            "typescript": {
                "cmd": ["typescript-language-server", "--stdio"],
                "extensions": [".ts", ".tsx", ".js", ".jsx"],
                "language_id": "typescript"
            },
            "rust": {
                "cmd": ["rust-analyzer"],
                "extensions": [".rs"],
                "language_id": "rust"
            },
            "go": {
                "cmd": ["gopls"],
                "extensions": [".go"],
                "language_id": "go"
            }
        }
        
        logger.info("Code Intelligence Service initialized")
    
    def _map_project_path(self, project_path: str) -> str:
        """Map external paths to container paths"""
        for external_path, container_path in self.path_mappings.items():
            if project_path.startswith(external_path):
                mapped_path = project_path.replace(external_path, container_path)
                logger.info(f"Mapped path: {project_path} -> {mapped_path}")
                return mapped_path
        return project_path
    
    async def activate_project(self, project_path: str) -> ProjectContext:
        """Activate a project and initialize its context"""
        # Map the path to container path
        mapped_path = self._map_project_path(project_path)
        project_path = Path(mapped_path).resolve()
        project_key = str(project_path)
        
        if project_key in self.projects:
            logger.info(f"Project already active: {project_path}")
            return self.projects[project_key]
        
        # Detect project language and framework
        language = self._detect_language(project_path)
        framework = self._detect_framework(project_path)
        
        # Create project context
        context = ProjectContext(
            project_root=str(project_path),
            language=language,
            framework=framework
        )
        
        # Load dependencies
        context.dependencies = await self._load_dependencies(project_path, language)
        
        # Initialize LSP workspace
        await self._initialize_lsp(project_path, language)
        
        # Load project memories if they exist
        await self._load_project_memories(context)
        
        self.projects[project_key] = context
        logger.info(f"Project activated: {project_path} (lang={language}, framework={framework})")
        
        return context
    
    def _detect_language(self, project_path: Path) -> str:
        """Detect primary language of project"""
        language_files = {
            "python": ["*.py", "requirements.txt", "setup.py", "pyproject.toml"],
            "typescript": ["*.ts", "*.tsx", "package.json", "tsconfig.json"],
            "javascript": ["*.js", "*.jsx", "package.json"],
            "rust": ["*.rs", "Cargo.toml"],
            "go": ["*.go", "go.mod"]
        }
        
        for language, patterns in language_files.items():
            for pattern in patterns:
                if list(project_path.rglob(pattern)):
                    return language
        
        return "unknown"
    
    def _detect_framework(self, project_path: Path) -> Optional[str]:
        """Detect framework used in project"""
        framework_indicators = {
            "fastapi": ["from fastapi", "FastAPI()"],
            "django": ["django", "manage.py", "settings.py"],
            "flask": ["from flask", "Flask()"],
            "react": ["react", "ReactDOM", "jsx", "tsx"],
            "vue": ["vue", ".vue", "Vue."],
            "angular": ["@angular", "angular.json"],
            "nextjs": ["next.config", "pages/", "_app."],
            "express": ["express", "app.listen"],
        }
        
        # Check common files for framework indicators
        for framework, indicators in framework_indicators.items():
            for indicator in indicators:
                # Check in common files
                for file_path in project_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in [".py", ".js", ".ts", ".tsx", ".jsx"]:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read(1000)  # Read first 1KB
                                if indicator in content:
                                    return framework
                        except:
                            continue
        
        return None
    
    async def _load_dependencies(self, project_path: Path, language: str) -> List[str]:
        """Load project dependencies"""
        dependencies = []
        
        if language == "python":
            req_file = project_path / "requirements.txt"
            if req_file.exists():
                with open(req_file, 'r') as f:
                    dependencies = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            
            pyproject = project_path / "pyproject.toml"
            if pyproject.exists():
                try:
                    import toml
                    data = toml.load(pyproject)
                    if "tool" in data and "poetry" in data["tool"]:
                        deps = data["tool"]["poetry"].get("dependencies", {})
                        dependencies.extend(deps.keys())
                except:
                    pass
        
        elif language in ["typescript", "javascript"]:
            package_json = project_path / "package.json"
            if package_json.exists():
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    dependencies = list(data.get("dependencies", {}).keys())
                    dependencies.extend(data.get("devDependencies", {}).keys())
        
        return dependencies
    
    async def _initialize_lsp(self, project_path: Path, language: str):
        """Initialize Language Server Protocol for project"""
        if language not in self.lsp_configs:
            logger.warning(f"No LSP configuration for language: {language}")
            return
        
        project_key = str(project_path)
        
        # Create workspace
        workspace = Workspace(str(project_path), None)
        self.workspaces[project_key] = workspace
        
        logger.info(f"LSP workspace initialized for {project_path}")
    
    async def _load_project_memories(self, context: ProjectContext):
        """Load project-specific memories"""
        project_hash = hashlib.md5(context.project_root.encode()).hexdigest()[:8]
        memory_path = self.memory_dir / f"project_{project_hash}"
        
        if memory_path.exists():
            for memory_file in memory_path.glob("*.md"):
                with open(memory_file, 'r') as f:
                    context.memory_items[memory_file.stem] = f.read()
            
            logger.info(f"Loaded {len(context.memory_items)} memories for project")
    
    async def get_symbols_overview(self, project_path: str, file_path: str = None) -> List[Dict]:
        """Get overview of symbols in a file or project"""
        # Map the path to container path
        mapped_path = self._map_project_path(project_path)
        context = self.projects.get(str(Path(mapped_path).resolve()))
        if not context:
            context = await self.activate_project(project_path)
        
        symbols = []
        
        if file_path:
            # Get symbols for specific file
            symbols = await self._analyze_file_symbols(Path(file_path), context)
        else:
            # Get overview of entire project
            project_root = Path(context.project_root)
            for ext in self._get_extensions_for_language(context.language):
                for file_path in project_root.rglob(f"*{ext}"):
                    if not any(part.startswith('.') for part in file_path.parts):
                        file_symbols = await self._analyze_file_symbols(file_path, context)
                        symbols.extend(file_symbols)
        
        return [s.to_dict() for s in symbols]
    
    async def _analyze_file_symbols(self, file_path: Path, context: ProjectContext) -> List[CodeSymbol]:
        """Analyze symbols in a single file"""
        symbols = []
        
        if context.language == "python":
            import ast
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        symbol = CodeSymbol(
                            name=node.name,
                            kind="class",
                            path=str(file_path),
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            name_path=node.name,
                            docstring=ast.get_docstring(node)
                        )
                        
                        # Add methods as children
                        children = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                child = CodeSymbol(
                                    name=item.name,
                                    kind="method",
                                    path=str(file_path),
                                    line_start=item.lineno,
                                    line_end=item.end_lineno or item.lineno,
                                    name_path=f"{node.name}/{item.name}",
                                    docstring=ast.get_docstring(item)
                                )
                                children.append(child)
                        symbol.children = children
                        symbols.append(symbol)
                    
                    elif isinstance(node, ast.FunctionDef) and not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                        symbol = CodeSymbol(
                            name=node.name,
                            kind="function",
                            path=str(file_path),
                            line_start=node.lineno,
                            line_end=node.end_lineno or node.lineno,
                            name_path=node.name,
                            docstring=ast.get_docstring(node)
                        )
                        symbols.append(symbol)
            except:
                logger.error(f"Failed to parse {file_path}")
        
        return symbols
    
    def _get_extensions_for_language(self, language: str) -> List[str]:
        """Get file extensions for a language"""
        extensions_map = {
            "python": [".py"],
            "typescript": [".ts", ".tsx"],
            "javascript": [".js", ".jsx"],
            "rust": [".rs"],
            "go": [".go"]
        }
        return extensions_map.get(language, [])
    
    async def find_symbol(
        self,
        project_path: str,
        name_path: str,
        include_body: bool = False,
        include_references: bool = False
    ) -> Optional[Dict]:
        """Find a symbol by name path"""
        context = self.projects.get(project_path)
        if not context:
            context = await self.activate_project(project_path)
        
        # Search in cache first
        if name_path in context.symbols_cache:
            symbol = context.symbols_cache[name_path]
            return symbol.to_dict()
        
        # Search in project
        symbols = await self.get_symbols_overview(project_path)
        
        for symbol_dict in symbols:
            if symbol_dict["name_path"] == name_path or symbol_dict["name"] == name_path:
                if include_body:
                    # Load the actual code
                    symbol_dict["body"] = await self._get_symbol_body(symbol_dict)
                
                if include_references:
                    # Find references to this symbol
                    symbol_dict["references"] = await self._find_references(project_path, name_path)
                
                return symbol_dict
            
            # Check children
            for child in symbol_dict.get("children", []):
                if child["name_path"] == name_path:
                    if include_body:
                        child["body"] = await self._get_symbol_body(child)
                    if include_references:
                        child["references"] = await self._find_references(project_path, name_path)
                    return child
        
        return None
    
    async def _get_symbol_body(self, symbol: Dict) -> str:
        """Get the body/source code of a symbol"""
        try:
            with open(symbol["path"], 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start = symbol["line_start"] - 1
                end = symbol["line_end"]
                return ''.join(lines[start:end])
        except:
            return ""
    
    async def _find_references(self, project_path: str, symbol_name: str) -> List[Dict]:
        """Find all references to a symbol"""
        references = []
        project_root = Path(project_path)
        
        # Simple grep-based reference finding
        # In a real implementation, this would use LSP's find references
        try:
            result = subprocess.run(
                ["grep", "-rn", symbol_name, str(project_root), "--include=*.py"],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        references.append({
                            "file": parts[0],
                            "line": int(parts[1]),
                            "text": parts[2].strip()
                        })
        except:
            pass
        
        return references
    
    async def replace_symbol(
        self,
        project_path: str,
        symbol_path: str,
        new_body: str
    ) -> Dict:
        """Replace a symbol's body with new code"""
        symbol = await self.find_symbol(project_path, symbol_path, include_body=True)
        if not symbol:
            return {"success": False, "message": f"Symbol {symbol_path} not found"}
        
        try:
            # Read the file
            with open(symbol["path"], 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Calculate indentation
            start_line = lines[symbol["line_start"] - 1]
            indent = len(start_line) - len(start_line.lstrip())
            
            # Prepare new body with proper indentation
            new_lines = []
            for line in new_body.split('\n'):
                if line.strip():
                    new_lines.append(' ' * indent + line + '\n')
                else:
                    new_lines.append('\n')
            
            # Replace the symbol body
            lines[symbol["line_start"] - 1:symbol["line_end"]] = new_lines
            
            # Write back
            with open(symbol["path"], 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return {"success": True, "message": f"Symbol {symbol_path} updated"}
        
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    async def save_memory(self, project_path: str, name: str, content: str) -> Dict:
        """Save a project-specific memory"""
        context = self.projects.get(project_path)
        if not context:
            context = await self.activate_project(project_path)
        
        # Save to memory
        context.memory_items[name] = content
        
        # Persist to disk
        project_hash = hashlib.md5(context.project_root.encode()).hexdigest()[:8]
        memory_path = self.memory_dir / f"project_{project_hash}"
        memory_path.mkdir(exist_ok=True)
        
        memory_file = memory_path / f"{name}.md"
        with open(memory_file, 'w') as f:
            f.write(content)
        
        return {"success": True, "message": f"Memory '{name}' saved"}
    
    async def load_memory(self, project_path: str, name: str) -> Optional[str]:
        """Load a project-specific memory"""
        context = self.projects.get(project_path)
        if not context:
            context = await self.activate_project(project_path)
        
        return context.memory_items.get(name)
    
    async def list_memories(self, project_path: str) -> List[str]:
        """List all memories for a project"""
        context = self.projects.get(project_path)
        if not context:
            context = await self.activate_project(project_path)
        
        return list(context.memory_items.keys())
    
    async def search_pattern(
        self,
        project_path: str,
        pattern: str,
        file_pattern: str = None,
        context_lines: int = 2
    ) -> List[Dict]:
        """Search for a pattern in project files"""
        results = []
        project_root = Path(project_path)
        
        # Build grep command
        cmd = ["grep", "-rn", "--color=never"]
        if context_lines > 0:
            cmd.extend([f"-C{context_lines}"])
        
        if file_pattern:
            cmd.extend(["--include", file_pattern])
        
        cmd.extend([pattern, str(project_root)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            current_file = None
            current_matches = []
            
            for line in result.stdout.split('\n'):
                if '--' in line:
                    continue
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        file_path = parts[0]
                        line_num = parts[1]
                        text = parts[2]
                        
                        if current_file != file_path:
                            if current_file and current_matches:
                                results.append({
                                    "file": current_file,
                                    "matches": current_matches
                                })
                            current_file = file_path
                            current_matches = []
                        
                        current_matches.append({
                            "line": int(line_num) if line_num.isdigit() else 0,
                            "text": text
                        })
            
            if current_file and current_matches:
                results.append({
                    "file": current_file,
                    "matches": current_matches
                })
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
        
        return results


# Singleton instance
code_intelligence_service = CodeIntelligenceService()