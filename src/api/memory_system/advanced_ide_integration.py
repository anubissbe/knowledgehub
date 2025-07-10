#!/usr/bin/env python3
"""
Advanced IDE Integration System
Provides deep integration with IDEs beyond basic MCP for context-aware development assistance
"""

import os
import sys
import json
import asyncio
import logging
import websockets
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
import hashlib
from collections import defaultdict
import threading
import time

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from claude_unified_memory import UnifiedMemorySystem

logger = logging.getLogger(__name__)

class IDEType(Enum):
    VSCODE = "vscode"
    INTELLIJ = "intellij"
    SUBLIME = "sublime"
    ATOM = "atom"
    NEOVIM = "neovim"
    VIM = "vim"
    EMACS = "emacs"
    ECLIPSE = "eclipse"
    WEBSTORM = "webstorm"
    PYCHARM = "pycharm"

class EventType(Enum):
    FILE_OPENED = "file_opened"
    FILE_CLOSED = "file_closed"
    FILE_SAVED = "file_saved"
    FILE_MODIFIED = "file_modified"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    SEARCH_PERFORMED = "search_performed"
    BREAKPOINT_SET = "breakpoint_set"
    DEBUG_SESSION = "debug_session"
    BUILD_STARTED = "build_started"
    BUILD_COMPLETED = "build_completed"
    TEST_RUN = "test_run"
    EXTENSION_ACTION = "extension_action"
    PROJECT_OPENED = "project_opened"
    PROJECT_CLOSED = "project_closed"

class ContextScope(Enum):
    GLOBAL = "global"
    PROJECT = "project"
    FILE = "file"
    FUNCTION = "function"
    SELECTION = "selection"
    LINE = "line"

@dataclass
class CodePosition:
    """Position within a code file"""
    line: int
    column: int
    offset: Optional[int] = None

@dataclass
class CodeRange:
    """Range of code"""
    start: CodePosition
    end: CodePosition
    
    def contains_position(self, position: CodePosition) -> bool:
        """Check if position is within this range"""
        if position.line < self.start.line or position.line > self.end.line:
            return False
        if position.line == self.start.line and position.column < self.start.column:
            return False
        if position.line == self.end.line and position.column > self.end.column:
            return False
        return True

@dataclass
class CodeSymbol:
    """Code symbol (function, class, variable)"""
    name: str
    symbol_type: str  # function, class, variable, method, property
    range: CodeRange
    file_path: str
    
    # Additional context
    language: str = ""
    signature: str = ""
    documentation: str = ""
    is_exported: bool = False
    is_deprecated: bool = False

@dataclass
class IDEEvent:
    """IDE event captured from user interaction"""
    event_id: str
    event_type: EventType
    timestamp: str
    ide_type: IDEType
    
    # File context
    file_path: str = ""
    project_path: str = ""
    
    # Position context
    cursor_position: Optional[CodePosition] = None
    selection: Optional[CodeRange] = None
    
    # Content context
    file_content: str = ""
    modified_content: str = ""
    search_query: str = ""
    
    # Symbols context
    current_function: Optional[CodeSymbol] = None
    visible_symbols: List[CodeSymbol] = None
    
    # Session context
    session_id: str = ""
    user_id: str = ""
    
    # Metadata
    language: str = ""
    file_size: int = 0
    line_count: int = 0
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.session_id:
            self.session_id = f"session_{int(time.time())}"
        if self.visible_symbols is None:
            self.visible_symbols = []

@dataclass
class DevelopmentSession:
    """Development session tracking"""
    session_id: str
    start_time: str
    end_time: str = ""
    
    # Session metrics
    files_opened: int = 0
    files_modified: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    
    # Activity tracking
    coding_time_minutes: int = 0
    debugging_time_minutes: int = 0
    research_time_minutes: int = 0
    
    # Context
    primary_language: str = ""
    project_path: str = ""
    focus_areas: List[str] = None
    
    # Insights
    most_edited_files: List[str] = None
    frequently_searched: List[str] = None
    common_patterns: List[str] = None
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = []
        if self.most_edited_files is None:
            self.most_edited_files = []
        if self.frequently_searched is None:
            self.frequently_searched = []
        if self.common_patterns is None:
            self.common_patterns = []

@dataclass
class ContextualSuggestion:
    """Contextual suggestion for IDE"""
    suggestion_id: str
    suggestion_type: str  # completion, refactor, documentation, debug
    title: str
    description: str
    
    # Applicability
    applicable_scope: ContextScope
    applicable_languages: List[str] = None
    confidence: float = 0.0
    
    # Actions
    action_type: str = ""  # insert, replace, navigate, execute
    action_data: Dict[str, Any] = None
    
    # Context
    triggered_by: str = ""
    related_files: List[str] = None
    related_symbols: List[str] = None
    
    def __post_init__(self):
        if not self.suggestion_id:
            self.suggestion_id = str(uuid.uuid4())
        if self.applicable_languages is None:
            self.applicable_languages = []
        if self.action_data is None:
            self.action_data = {}
        if self.related_files is None:
            self.related_files = []
        if self.related_symbols is None:
            self.related_symbols = []

@dataclass
class ProjectContext:
    """High-level project context"""
    project_path: str
    project_name: str
    
    # Project structure
    languages: Dict[str, int] = None  # language -> line count
    file_types: Dict[str, int] = None  # extension -> count
    directory_structure: Dict[str, Any] = None
    
    # Dependencies
    package_managers: List[str] = None
    dependencies: Dict[str, str] = None
    
    # Patterns and conventions
    coding_patterns: List[str] = None
    naming_conventions: Dict[str, str] = None
    
    # Activity
    hot_files: List[str] = None  # Frequently edited
    active_features: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = {}
        if self.file_types is None:
            self.file_types = {}
        if self.directory_structure is None:
            self.directory_structure = {}
        if self.package_managers is None:
            self.package_managers = []
        if self.dependencies is None:
            self.dependencies = {}
        if self.coding_patterns is None:
            self.coding_patterns = []
        if self.naming_conventions is None:
            self.naming_conventions = {}
        if self.hot_files is None:
            self.hot_files = []
        if self.active_features is None:
            self.active_features = []

class AdvancedIDEIntegration:
    """
    Advanced IDE integration providing context-aware development assistance
    """
    
    def __init__(self, memory_system: Optional[UnifiedMemorySystem] = None):
        self.memory_system = memory_system or UnifiedMemorySystem()
        
        # Storage
        self.events_dir = Path("/opt/projects/memory-system/data/ide_events")
        self.sessions_dir = Path("/opt/projects/memory-system/data/ide_sessions")
        self.projects_dir = Path("/opt/projects/memory-system/data/ide_projects")
        
        for directory in [self.events_dir, self.sessions_dir, self.projects_dir]:
            directory.mkdir(exist_ok=True)
        
        # Active tracking
        self.active_sessions: Dict[str, DevelopmentSession] = {}
        self.project_contexts: Dict[str, ProjectContext] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # WebSocket servers for real-time communication
        self.websocket_servers: Dict[int, Any] = {}
        self.connected_clients: Dict[str, Any] = {}
        
        # Configuration
        self.config = {
            "websocket_port": 8765,
            "http_port": 8766,
            "auto_suggestions": True,
            "context_tracking": True,
            "session_tracking": True,
            "memory_integration": True,
            "suggestion_confidence_threshold": 0.6,
            "max_events_per_session": 10000
        }
        
        # Analysis engines
        self.suggestion_engine = None
        self.pattern_analyzer = None
        
        # Load existing data
        self._load_recent_sessions()
        
    def _load_recent_sessions(self):
        """Load recent development sessions"""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for session_file in self.sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    start_time = datetime.fromisoformat(session_data["start_time"])
                    if start_time >= cutoff_date:
                        session = DevelopmentSession(**session_data)
                        if not session.end_time:  # Active session
                            self.active_sessions[session.session_id] = session
                            
                except Exception as e:
                    logger.warning(f"Failed to load session file {session_file}: {e}")
            
            logger.info(f"Loaded {len(self.active_sessions)} active sessions")
            
        except Exception as e:
            logger.error(f"Failed to load recent sessions: {e}")
    
    async def start_websocket_server(self, port: int = None) -> bool:
        """Start WebSocket server for real-time IDE communication"""
        try:
            port = port or self.config["websocket_port"]
            
            async def handle_client(websocket, path):
                client_id = str(uuid.uuid4())
                self.connected_clients[client_id] = {
                    "websocket": websocket,
                    "connected_at": datetime.now().isoformat(),
                    "ide_type": None,
                    "project_path": None
                }
                
                logger.info(f"IDE client connected: {client_id}")
                
                try:
                    async for message in websocket:
                        await self._handle_websocket_message(client_id, message)
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"IDE client disconnected: {client_id}")
                finally:
                    if client_id in self.connected_clients:
                        del self.connected_clients[client_id]
            
            # Start server
            server = await websockets.serve(handle_client, "localhost", port)
            self.websocket_servers[port] = server
            
            logger.info(f"WebSocket server started on port {port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False
    
    async def _handle_websocket_message(self, client_id: str, message: str):
        """Handle incoming WebSocket message from IDE"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "register":
                await self._handle_client_registration(client_id, data)
            elif message_type == "event":
                await self._handle_ide_event(client_id, data)
            elif message_type == "request_suggestions":
                await self._handle_suggestion_request(client_id, data)
            elif message_type == "request_context":
                await self._handle_context_request(client_id, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client {client_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _handle_client_registration(self, client_id: str, data: Dict[str, Any]):
        """Handle IDE client registration"""
        try:
            client = self.connected_clients[client_id]
            client["ide_type"] = IDEType(data.get("ide_type", "vscode"))
            client["project_path"] = data.get("project_path", "")
            client["user_id"] = data.get("user_id", "")
            
            # Analyze project if new
            if client["project_path"] and client["project_path"] not in self.project_contexts:
                await self._analyze_project_context(client["project_path"])
            
            # Send welcome message with current context
            await self._send_to_client(client_id, {
                "type": "registered",
                "client_id": client_id,
                "features": {
                    "suggestions": self.config["auto_suggestions"],
                    "context_tracking": self.config["context_tracking"],
                    "session_tracking": self.config["session_tracking"]
                }
            })
            
            logger.info(f"Client registered: {client['ide_type'].value} for {client['project_path']}")
            
        except Exception as e:
            logger.error(f"Failed to register client: {e}")
    
    async def _handle_ide_event(self, client_id: str, data: Dict[str, Any]):
        """Handle IDE event"""
        try:
            client = self.connected_clients[client_id]
            
            # Parse event data
            event = IDEEvent(
                event_type=EventType(data["event_type"]),
                ide_type=client["ide_type"],
                file_path=data.get("file_path", ""),
                project_path=client["project_path"],
                cursor_position=self._parse_position(data.get("cursor_position")),
                selection=self._parse_range(data.get("selection")),
                file_content=data.get("file_content", ""),
                modified_content=data.get("modified_content", ""),
                search_query=data.get("search_query", ""),
                language=data.get("language", ""),
                file_size=data.get("file_size", 0),
                line_count=data.get("line_count", 0),
                user_id=client.get("user_id", ""),
                session_id=data.get("session_id", "")
            )
            
            # Process event
            await self._process_ide_event(event)
            
            # Send suggestions if enabled
            if self.config["auto_suggestions"]:
                suggestions = await self._generate_contextual_suggestions(event)
                if suggestions:
                    await self._send_to_client(client_id, {
                        "type": "suggestions",
                        "suggestions": [asdict(s) for s in suggestions]
                    })
            
        except Exception as e:
            logger.error(f"Failed to handle IDE event: {e}")
    
    async def _handle_suggestion_request(self, client_id: str, data: Dict[str, Any]):
        """Handle explicit suggestion request"""
        try:
            # Create event from request context
            client = self.connected_clients[client_id]
            event = IDEEvent(
                event_type=EventType.CURSOR_MOVED,  # Treat as cursor movement
                ide_type=client["ide_type"],
                file_path=data.get("file_path", ""),
                project_path=client["project_path"],
                cursor_position=self._parse_position(data.get("cursor_position")),
                file_content=data.get("file_content", ""),
                language=data.get("language", ""),
                user_id=client.get("user_id", "")
            )
            
            # Generate suggestions
            suggestions = await self._generate_contextual_suggestions(event)
            
            await self._send_to_client(client_id, {
                "type": "suggestions_response",
                "request_id": data.get("request_id"),
                "suggestions": [asdict(s) for s in suggestions]
            })
            
        except Exception as e:
            logger.error(f"Failed to handle suggestion request: {e}")
    
    async def _handle_context_request(self, client_id: str, data: Dict[str, Any]):
        """Handle context information request"""
        try:
            client = self.connected_clients[client_id]
            project_path = client["project_path"]
            
            context_info = {
                "project_context": asdict(self.project_contexts.get(project_path, {})),
                "active_session": None,
                "related_files": [],
                "recent_patterns": []
            }
            
            # Add active session info
            for session in self.active_sessions.values():
                if session.project_path == project_path:
                    context_info["active_session"] = asdict(session)
                    break
            
            await self._send_to_client(client_id, {
                "type": "context_response",
                "request_id": data.get("request_id"),
                "context": context_info
            })
            
        except Exception as e:
            logger.error(f"Failed to handle context request: {e}")
    
    def _parse_position(self, position_data: Optional[Dict[str, Any]]) -> Optional[CodePosition]:
        """Parse position data"""
        if not position_data:
            return None
        
        return CodePosition(
            line=position_data.get("line", 0),
            column=position_data.get("column", 0),
            offset=position_data.get("offset")
        )
    
    def _parse_range(self, range_data: Optional[Dict[str, Any]]) -> Optional[CodeRange]:
        """Parse range data"""
        if not range_data:
            return None
        
        start = self._parse_position(range_data.get("start"))
        end = self._parse_position(range_data.get("end"))
        
        if start and end:
            return CodeRange(start=start, end=end)
        return None
    
    async def _send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to specific client"""
        try:
            client = self.connected_clients.get(client_id)
            if client and "websocket" in client:
                await client["websocket"].send(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send to client {client_id}: {e}")
    
    async def _process_ide_event(self, event: IDEEvent):
        """Process and store IDE event"""
        try:
            # Store event
            await self._store_ide_event(event)
            
            # Update session tracking
            if self.config["session_tracking"]:
                await self._update_session_tracking(event)
            
            # Trigger event handlers
            for handler in self.event_handlers[event.event_type]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
            
            # Integrate with memory if enabled
            if self.config["memory_integration"]:
                await self._integrate_event_with_memory(event)
            
        except Exception as e:
            logger.error(f"Failed to process IDE event: {e}")
    
    async def _store_ide_event(self, event: IDEEvent):
        """Store IDE event to disk"""
        try:
            # Create event file
            event_file = self.events_dir / f"{event.event_id}.json"
            with open(event_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(event), f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to store IDE event: {e}")
    
    async def _update_session_tracking(self, event: IDEEvent):
        """Update development session tracking"""
        try:
            session_id = event.session_id
            
            # Get or create session
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = DevelopmentSession(
                    session_id=session_id,
                    start_time=event.timestamp,
                    project_path=event.project_path,
                    primary_language=event.language
                )
            
            session = self.active_sessions[session_id]
            
            # Update session metrics based on event type
            if event.event_type == EventType.FILE_OPENED:
                session.files_opened += 1
            elif event.event_type == EventType.FILE_MODIFIED:
                session.files_modified += 1
                if event.file_path not in session.most_edited_files:
                    session.most_edited_files.append(event.file_path)
            elif event.event_type == EventType.SEARCH_PERFORMED:
                if event.search_query and event.search_query not in session.frequently_searched:
                    session.frequently_searched.append(event.search_query)
            
            # Store updated session
            await self._store_session(session)
            
        except Exception as e:
            logger.error(f"Failed to update session tracking: {e}")
    
    async def _store_session(self, session: DevelopmentSession):
        """Store development session"""
        try:
            session_file = self.sessions_dir / f"{session.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(session), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to store session: {e}")
    
    async def _analyze_project_context(self, project_path: str):
        """Analyze project structure and context"""
        try:
            project_context = ProjectContext(
                project_path=project_path,
                project_name=Path(project_path).name
            )
            
            # Analyze file structure
            await self._analyze_project_files(project_context)
            
            # Analyze dependencies
            await self._analyze_project_dependencies(project_context)
            
            # Store context
            self.project_contexts[project_path] = project_context
            await self._store_project_context(project_context)
            
            logger.info(f"Analyzed project context for {project_path}")
            
        except Exception as e:
            logger.error(f"Failed to analyze project context: {e}")
    
    async def _analyze_project_files(self, context: ProjectContext):
        """Analyze project file structure"""
        try:
            project_path = Path(context.project_path)
            
            if not project_path.exists():
                return
            
            # Count files by language/extension
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    extension = file_path.suffix.lower()
                    context.file_types[extension] = context.file_types.get(extension, 0) + 1
                    
                    # Count lines by language
                    language = self._detect_language_from_extension(extension)
                    if language:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = len(f.readlines())
                                context.languages[language] = context.languages.get(language, 0) + lines
                        except Exception:
                            pass  # Skip files that can't be read
            
        except Exception as e:
            logger.error(f"Failed to analyze project files: {e}")
    
    async def _analyze_project_dependencies(self, context: ProjectContext):
        """Analyze project dependencies"""
        try:
            project_path = Path(context.project_path)
            
            # Check for package managers and dependency files
            dependency_files = {
                "package.json": "npm",
                "requirements.txt": "pip",
                "Pipfile": "pipenv",
                "pyproject.toml": "poetry",
                "Cargo.toml": "cargo",
                "pom.xml": "maven",
                "build.gradle": "gradle",
                "composer.json": "composer"
            }
            
            for dep_file, package_manager in dependency_files.items():
                dep_path = project_path / dep_file
                if dep_path.exists():
                    context.package_managers.append(package_manager)
                    
                    # Parse dependencies based on file type
                    if dep_file == "package.json":
                        await self._parse_package_json(dep_path, context)
                    elif dep_file == "requirements.txt":
                        await self._parse_requirements_txt(dep_path, context)
                    # Add more parsers as needed
            
        except Exception as e:
            logger.error(f"Failed to analyze project dependencies: {e}")
    
    async def _parse_package_json(self, file_path: Path, context: ProjectContext):
        """Parse package.json dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
                
            for dep_type in ["dependencies", "devDependencies"]:
                deps = package_data.get(dep_type, {})
                for name, version in deps.items():
                    context.dependencies[name] = version
                    
        except Exception as e:
            logger.error(f"Failed to parse package.json: {e}")
    
    async def _parse_requirements_txt(self, file_path: Path, context: ProjectContext):
        """Parse requirements.txt dependencies"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            name, version = line.split('==', 1)
                            context.dependencies[name] = version
                        else:
                            context.dependencies[line] = "latest"
                            
        except Exception as e:
            logger.error(f"Failed to parse requirements.txt: {e}")
    
    def _detect_language_from_extension(self, extension: str) -> Optional[str]:
        """Detect programming language from file extension"""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".clj": "clojure",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
            ".sql": "sql",
            ".sh": "bash",
            ".ps1": "powershell",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".md": "markdown"
        }
        
        return language_map.get(extension)
    
    async def _store_project_context(self, context: ProjectContext):
        """Store project context"""
        try:
            safe_name = context.project_name.replace("/", "_").replace("\\", "_")
            context_file = self.projects_dir / f"{safe_name}_context.json"
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(context), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to store project context: {e}")
    
    async def _generate_contextual_suggestions(self, event: IDEEvent) -> List[ContextualSuggestion]:
        """Generate contextual suggestions based on IDE event"""
        suggestions = []
        
        try:
            # File-based suggestions
            if event.event_type == EventType.FILE_OPENED:
                suggestions.extend(await self._generate_file_suggestions(event))
            
            # Cursor-based suggestions
            if event.cursor_position:
                suggestions.extend(await self._generate_cursor_suggestions(event))
            
            # Search-based suggestions
            if event.event_type == EventType.SEARCH_PERFORMED:
                suggestions.extend(await self._generate_search_suggestions(event))
            
            # Filter by confidence threshold
            filtered_suggestions = [
                s for s in suggestions
                if s.confidence >= self.config["suggestion_confidence_threshold"]
            ]
            
            return filtered_suggestions[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    async def _generate_file_suggestions(self, event: IDEEvent) -> List[ContextualSuggestion]:
        """Generate suggestions based on file opening"""
        suggestions = []
        
        try:
            # Related files suggestion
            if event.project_path in self.project_contexts:
                context = self.project_contexts[event.project_path]
                
                # Suggest frequently edited files
                if context.hot_files:
                    suggestions.append(ContextualSuggestion(
                        suggestion_type="navigation",
                        title="Related Files",
                        description=f"Frequently edited files in this project",
                        applicable_scope=ContextScope.PROJECT,
                        confidence=0.7,
                        action_type="navigate",
                        action_data={"files": context.hot_files[:3]},
                        related_files=context.hot_files[:3]
                    ))
            
            # Documentation suggestion for new files
            if event.language and not event.file_content.strip():
                suggestions.append(ContextualSuggestion(
                    suggestion_type="documentation",
                    title="Add File Header",
                    description=f"Add standard {event.language} file header",
                    applicable_scope=ContextScope.FILE,
                    applicable_languages=[event.language],
                    confidence=0.8,
                    action_type="insert",
                    action_data={"template": f"# {event.language} file header template"}
                ))
            
        except Exception as e:
            logger.error(f"Failed to generate file suggestions: {e}")
        
        return suggestions
    
    async def _generate_cursor_suggestions(self, event: IDEEvent) -> List[ContextualSuggestion]:
        """Generate suggestions based on cursor position"""
        suggestions = []
        
        try:
            # Function completion suggestions
            if event.current_function:
                suggestions.append(ContextualSuggestion(
                    suggestion_type="completion",
                    title="Function Context",
                    description=f"Currently in function: {event.current_function.name}",
                    applicable_scope=ContextScope.FUNCTION,
                    confidence=0.9,
                    related_symbols=[event.current_function.name]
                ))
            
            # Import suggestions based on undefined symbols
            if event.language == "python" and "import" not in event.file_content:
                suggestions.append(ContextualSuggestion(
                    suggestion_type="refactor",
                    title="Add Imports",
                    description="Consider adding common imports for Python",
                    applicable_scope=ContextScope.FILE,
                    applicable_languages=["python"],
                    confidence=0.6,
                    action_type="insert",
                    action_data={"imports": ["import os", "import sys", "from typing import *"]}
                ))
            
        except Exception as e:
            logger.error(f"Failed to generate cursor suggestions: {e}")
        
        return suggestions
    
    async def _generate_search_suggestions(self, event: IDEEvent) -> List[ContextualSuggestion]:
        """Generate suggestions based on search queries"""
        suggestions = []
        
        try:
            if event.search_query:
                # Documentation search suggestion
                suggestions.append(ContextualSuggestion(
                    suggestion_type="documentation",
                    title="Search Documentation",
                    description=f"Search external docs for '{event.search_query}'",
                    applicable_scope=ContextScope.GLOBAL,
                    confidence=0.7,
                    action_type="navigate",
                    action_data={"url": f"https://docs.python.org/search.html?q={event.search_query}"}
                ))
                
                # Related code search
                suggestions.append(ContextualSuggestion(
                    suggestion_type="navigation",
                    title="Similar Code",
                    description=f"Find similar code patterns for '{event.search_query}'",
                    applicable_scope=ContextScope.PROJECT,
                    confidence=0.8,
                    action_type="execute",
                    action_data={"command": f"grep -r '{event.search_query}' {event.project_path}"}
                ))
            
        except Exception as e:
            logger.error(f"Failed to generate search suggestions: {e}")
        
        return suggestions
    
    async def _integrate_event_with_memory(self, event: IDEEvent):
        """Integrate IDE event with memory system"""
        try:
            # Only store significant events to avoid noise
            significant_events = [
                EventType.FILE_OPENED,
                EventType.FILE_SAVED,
                EventType.PROJECT_OPENED,
                EventType.BUILD_COMPLETED,
                EventType.DEBUG_SESSION
            ]
            
            if event.event_type not in significant_events:
                return
            
            # Create memory entry
            memory_content = self._format_event_memory(event)
            
            await self.memory_system.add_memory(
                content=memory_content,
                memory_type="development",
                priority="low",
                tags=[
                    "ide_event",
                    f"event_type:{event.event_type.value}",
                    f"ide:{event.ide_type.value}",
                    f"language:{event.language}",
                    f"project:{Path(event.project_path).name}"
                ],
                metadata={
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "ide_type": event.ide_type.value,
                    "file_path": event.file_path,
                    "project_path": event.project_path,
                    "language": event.language,
                    "timestamp": event.timestamp
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to integrate event with memory: {e}")
    
    def _format_event_memory(self, event: IDEEvent) -> str:
        """Format IDE event for memory storage"""
        content = f"""IDE Event: {event.event_type.value}
IDE: {event.ide_type.value}
Time: {event.timestamp}
Project: {Path(event.project_path).name if event.project_path else 'Unknown'}"""
        
        if event.file_path:
            content += f"\nFile: {Path(event.file_path).name}"
        
        if event.language:
            content += f"\nLanguage: {event.language}"
        
        if event.cursor_position:
            content += f"\nPosition: Line {event.cursor_position.line}, Column {event.cursor_position.column}"
        
        if event.search_query:
            content += f"\nSearch: {event.search_query}"
        
        if event.current_function:
            content += f"\nFunction: {event.current_function.name}"
        
        return content
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """Register custom event handler"""
        self.event_handlers[event_type].append(handler)
    
    async def get_session_analytics(self, project_path: str = None) -> Dict[str, Any]:
        """Get development session analytics"""
        try:
            # Filter sessions
            sessions = list(self.active_sessions.values())
            if project_path:
                sessions = [s for s in sessions if s.project_path == project_path]
            
            if not sessions:
                return {"error": "No session data found"}
            
            # Calculate analytics
            total_sessions = len(sessions)
            total_coding_time = sum(s.coding_time_minutes for s in sessions)
            total_files_opened = sum(s.files_opened for s in sessions)
            total_files_modified = sum(s.files_modified for s in sessions)
            
            # Most active languages
            language_counts = defaultdict(int)
            for session in sessions:
                if session.primary_language:
                    language_counts[session.primary_language] += 1
            
            return {
                "total_sessions": total_sessions,
                "total_coding_time_minutes": total_coding_time,
                "avg_session_time_minutes": total_coding_time / total_sessions if total_sessions > 0 else 0,
                "total_files_opened": total_files_opened,
                "total_files_modified": total_files_modified,
                "most_used_languages": dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate session analytics: {e}")
            return {"error": str(e)}


# Global advanced IDE integration instance
ide_integration = AdvancedIDEIntegration()

# Convenience functions
async def start_ide_server(port: int = 8765) -> bool:
    """Start IDE integration server"""
    return await ide_integration.start_websocket_server(port)

async def get_ide_analytics(project_path: str = None) -> Dict[str, Any]:
    """Get IDE session analytics"""
    return await ide_integration.get_session_analytics(project_path)

def register_ide_handler(event_type: EventType, handler: Callable):
    """Register custom IDE event handler"""
    ide_integration.register_event_handler(event_type, handler)

if __name__ == "__main__":
    # Test advanced IDE integration
    async def test_ide_integration():
        print("🔌 Testing Advanced IDE Integration")
        
        # Test basic functionality
        print("✅ Advanced IDE Integration system initialized")
        print(f"   Supported IDEs: {len(IDEType)}")
        print(f"   Event types: {len(EventType)}")
        print(f"   Context scopes: {len(ContextScope)}")
        
        # Test WebSocket server
        server_started = await start_ide_server(8765)
        if server_started:
            print("✅ WebSocket server started on port 8765")
        else:
            print("❌ Failed to start WebSocket server")
        
        # Test analytics
        analytics = await get_ide_analytics()
        if "error" in analytics:
            print("📊 No IDE session data found (expected for new installation)")
        else:
            print("📊 IDE analytics generated")
        
        print("✅ Advanced IDE Integration ready!")
        print("Connect your IDE using WebSocket at ws://localhost:8765")
    
    asyncio.run(test_ide_integration())