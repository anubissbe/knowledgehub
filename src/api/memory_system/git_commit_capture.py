#!/usr/bin/env python3
"""
Git Commit Context Capture System
Automatically captures development context from git commits, branches, and repository activity
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
import hashlib
from collections import defaultdict

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from claude_unified_memory import UnifiedMemorySystem

logger = logging.getLogger(__name__)

class CommitType(Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    CHORE = "chore"
    HOTFIX = "hotfix"
    RELEASE = "release"
    MERGE = "merge"
    REVERT = "revert"

class ChangeImpact(Enum):
    CRITICAL = "critical"      # Breaking changes, major features
    HIGH = "high"             # Important features, significant fixes
    MEDIUM = "medium"         # Standard changes, minor features
    LOW = "low"              # Documentation, tests, minor fixes
    MINIMAL = "minimal"       # Formatting, comments, trivial changes

@dataclass
class FileChange:
    """Represents a change to a single file"""
    file_path: str
    change_type: str  # "added", "modified", "deleted", "renamed"
    lines_added: int
    lines_deleted: int
    
    # Content analysis
    functions_changed: List[str] = None
    classes_changed: List[str] = None
    imports_changed: List[str] = None
    
    # Context
    language: str = ""
    file_size_kb: float = 0.0
    complexity_score: int = 0  # 1-10 scale
    
    def __post_init__(self):
        if self.functions_changed is None:
            self.functions_changed = []
        if self.classes_changed is None:
            self.classes_changed = []
        if self.imports_changed is None:
            self.imports_changed = []

@dataclass
class CommitContext:
    """Rich context captured from a git commit"""
    commit_hash: str
    commit_message: str
    author: str
    author_email: str
    timestamp: str
    
    # Classification
    commit_type: CommitType
    change_impact: ChangeImpact
    
    # Changes
    files_changed: List[FileChange]
    total_lines_added: int
    total_lines_deleted: int
    
    # Branch and repository context
    branch_name: str
    repository_path: str
    parent_commits: List[str] = None
    
    # Semantic analysis
    semantic_tags: List[str] = None
    related_issues: List[str] = None
    breaking_changes: List[str] = None
    
    # Integration context
    build_status: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None
    review_comments: List[str] = None
    
    # Memory context
    memory_id: str = ""
    context_id: str = ""
    
    def __post_init__(self):
        if self.parent_commits is None:
            self.parent_commits = []
        if self.semantic_tags is None:
            self.semantic_tags = []
        if self.related_issues is None:
            self.related_issues = []
        if self.breaking_changes is None:
            self.breaking_changes = []
        if self.review_comments is None:
            self.review_comments = []
        if not self.memory_id:
            self.memory_id = f"commit_{self.commit_hash[:12]}"
        if not self.context_id:
            self.context_id = str(uuid.uuid4())

@dataclass
class BranchContext:
    """Context about a git branch"""
    branch_name: str
    base_branch: str
    created_date: str
    last_activity: str
    
    # Branch metrics
    commit_count: int
    total_files_changed: int
    total_lines_changed: int
    
    # Development context
    feature_description: str = ""
    related_issues: List[str] = None
    merge_status: str = "active"  # active, merged, abandoned
    
    # Collaboration
    contributors: List[str] = None
    review_status: str = "in_progress"
    
    def __post_init__(self):
        if self.related_issues is None:
            self.related_issues = []
        if self.contributors is None:
            self.contributors = []

@dataclass
class RepositoryContext:
    """High-level repository context"""
    repository_path: str
    repository_name: str
    remote_url: str = ""
    
    # Current state
    current_branch: str = ""
    total_commits: int = 0
    total_branches: int = 0
    
    # Activity metrics
    recent_activity_score: float = 0.0
    hotspot_files: List[str] = None
    frequent_contributors: List[str] = None
    
    # Project context
    project_type: str = ""  # web, api, library, etc.
    technologies: List[str] = None
    dependencies: Dict[str, str] = None
    
    def __post_init__(self):
        if self.hotspot_files is None:
            self.hotspot_files = []
        if self.frequent_contributors is None:
            self.frequent_contributors = []
        if self.technologies is None:
            self.technologies = []
        if self.dependencies is None:
            self.dependencies = {}

class GitCommitCapture:
    """
    Captures and analyzes git commit context for development workflow integration
    """
    
    def __init__(self, memory_system: Optional[UnifiedMemorySystem] = None):
        self.memory_system = memory_system or UnifiedMemorySystem()
        
        # Storage
        self.commits_dir = Path("/opt/projects/memory-system/data/git_commits")
        self.branches_dir = Path("/opt/projects/memory-system/data/git_branches")
        self.repos_dir = Path("/opt/projects/memory-system/data/git_repositories")
        
        for directory in [self.commits_dir, self.branches_dir, self.repos_dir]:
            directory.mkdir(exist_ok=True)
        
        # Analysis patterns
        self.commit_patterns = self._initialize_commit_patterns()
        self.file_patterns = self._initialize_file_patterns()
        self.semantic_patterns = self._initialize_semantic_patterns()
        
        # Cached contexts
        self.repository_contexts: Dict[str, RepositoryContext] = {}
        self.branch_contexts: Dict[str, BranchContext] = {}
        
        # Configuration
        self.config = {
            "auto_capture_enabled": True,
            "capture_file_content": True,
            "max_file_size_kb": 1000,
            "analysis_depth": "full",  # minimal, standard, full
            "memory_integration": True,
            "retention_days": 90
        }
    
    def _initialize_commit_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for commit type classification"""
        return {
            "feature": [
                r"^feat(\([^)]+\))?:",
                r"^add\b",
                r"^implement\b",
                r"^create\b",
                r"new feature",
                r"feature:"
            ],
            "bugfix": [
                r"^fix(\([^)]+\))?:",
                r"^bug\b",
                r"^resolve\b",
                r"^correct\b",
                r"fix.*bug",
                r"resolve.*issue"
            ],
            "refactor": [
                r"^refactor(\([^)]+\))?:",
                r"^cleanup\b",
                r"^improve\b",
                r"^optimize\b",
                r"code cleanup",
                r"refactoring"
            ],
            "docs": [
                r"^docs(\([^)]+\))?:",
                r"^documentation\b",
                r"^readme\b",
                r"update.*docs",
                r"add.*documentation"
            ],
            "test": [
                r"^test(\([^)]+\))?:",
                r"^tests?\b",
                r"add.*test",
                r"test.*coverage",
                r"unit test"
            ],
            "chore": [
                r"^chore(\([^)]+\))?:",
                r"^build\b",
                r"^ci\b",
                r"^deps?\b",
                r"update.*dependencies",
                r"build system"
            ]
        }
    
    def _initialize_file_patterns(self) -> Dict[str, str]:
        """Initialize file type patterns"""
        return {
            r"\.py$": "python",
            r"\.js$|\.ts$|\.jsx$|\.tsx$": "javascript",
            r"\.java$": "java",
            r"\.cpp$|\.c$|\.h$": "c++",
            r"\.rs$": "rust",
            r"\.go$": "go",
            r"\.sql$": "sql",
            r"\.md$|\.rst$": "documentation",
            r"\.yml$|\.yaml$": "config",
            r"\.json$": "config",
            r"\.dockerfile$|Dockerfile": "docker",
            r"\.sh$|\.bash$": "shell"
        }
    
    def _initialize_semantic_patterns(self) -> Dict[str, List[str]]:
        """Initialize semantic analysis patterns"""
        return {
            "breaking_changes": [
                r"breaking change",
                r"BREAKING:",
                r"major version",
                r"api change",
                r"incompatible"
            ],
            "security": [
                r"security",
                r"vulnerability",
                r"CVE-",
                r"auth",
                r"permission",
                r"sanitiz"
            ],
            "performance": [
                r"performance",
                r"optimize",
                r"speed",
                r"memory",
                r"cache",
                r"latency"
            ],
            "integration": [
                r"integrate",
                r"connect",
                r"api",
                r"webhook",
                r"service"
            ]
        }
    
    async def capture_commit_context(
        self,
        repository_path: str,
        commit_hash: Optional[str] = None
    ) -> Optional[CommitContext]:
        """Capture context from a specific commit"""
        
        if not os.path.exists(repository_path):
            logger.error(f"Repository path does not exist: {repository_path}")
            return None
        
        try:
            # Get commit hash if not provided (use HEAD)
            if not commit_hash:
                commit_hash = await self._run_git_command(
                    repository_path, ["rev-parse", "HEAD"]
                )
                commit_hash = commit_hash.strip()
            
            # Get commit information
            commit_info = await self._get_commit_info(repository_path, commit_hash)
            if not commit_info:
                return None
            
            # Get file changes
            file_changes = await self._get_file_changes(repository_path, commit_hash)
            
            # Analyze commit
            commit_type = self._classify_commit_type(commit_info["message"])
            change_impact = self._assess_change_impact(file_changes, commit_info["message"])
            
            # Get branch context
            branch_name = await self._get_current_branch(repository_path)
            
            # Extract semantic information
            semantic_tags = self._extract_semantic_tags(commit_info["message"])
            related_issues = self._extract_issue_references(commit_info["message"])
            breaking_changes = self._extract_breaking_changes(commit_info["message"])
            
            # Create commit context
            context = CommitContext(
                commit_hash=commit_hash,
                commit_message=commit_info["message"],
                author=commit_info["author"],
                author_email=commit_info["email"],
                timestamp=commit_info["timestamp"],
                commit_type=commit_type,
                change_impact=change_impact,
                files_changed=file_changes,
                total_lines_added=sum(fc.lines_added for fc in file_changes),
                total_lines_deleted=sum(fc.lines_deleted for fc in file_changes),
                branch_name=branch_name,
                repository_path=repository_path,
                parent_commits=commit_info.get("parents", []),
                semantic_tags=semantic_tags,
                related_issues=related_issues,
                breaking_changes=breaking_changes
            )
            
            # Store context
            await self._store_commit_context(context)
            
            # Integrate with memory system
            if self.config["memory_integration"]:
                await self._integrate_with_memory(context)
            
            logger.info(f"Captured commit context: {commit_hash[:12]}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to capture commit context: {e}")
            return None
    
    async def _run_git_command(self, repo_path: str, args: List[str]) -> str:
        """Run a git command in the specified repository"""
        cmd = ["git", "-C", repo_path] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                logger.error(f"Git command failed: {' '.join(cmd)}\nError: {result.stderr}")
                return ""
        except subprocess.TimeoutExpired:
            logger.error(f"Git command timed out: {' '.join(cmd)}")
            return ""
        except Exception as e:
            logger.error(f"Error running git command: {e}")
            return ""
    
    async def _get_commit_info(self, repo_path: str, commit_hash: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a commit"""
        # Get commit details
        format_str = "%H%n%an%n%ae%n%at%n%P%n%s%n%b"
        output = await self._run_git_command(
            repo_path,
            ["show", "--format=" + format_str, "--no-patch", commit_hash]
        )
        
        if not output:
            return None
        
        lines = output.strip().split('\n')
        if len(lines) < 6:
            return None
        
        # Parse commit information
        return {
            "hash": lines[0],
            "author": lines[1],
            "email": lines[2],
            "timestamp": datetime.fromtimestamp(int(lines[3])).isoformat(),
            "parents": lines[4].split() if lines[4] else [],
            "subject": lines[5],
            "message": '\n'.join(lines[5:]).strip(),
            "body": '\n'.join(lines[6:]).strip() if len(lines) > 6 else ""
        }
    
    async def _get_file_changes(self, repo_path: str, commit_hash: str) -> List[FileChange]:
        """Get detailed file changes for a commit"""
        file_changes = []
        
        # Get file statistics
        stat_output = await self._run_git_command(
            repo_path,
            ["show", "--numstat", "--format=", commit_hash]
        )
        
        if not stat_output:
            return file_changes
        
        for line in stat_output.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            
            lines_added = int(parts[0]) if parts[0] != '-' else 0
            lines_deleted = int(parts[1]) if parts[1] != '-' else 0
            file_path = parts[2]
            
            # Determine change type
            change_type = await self._get_change_type(repo_path, commit_hash, file_path)
            
            # Analyze file content if enabled
            functions_changed = []
            classes_changed = []
            imports_changed = []
            
            if self.config["capture_file_content"]:
                content_analysis = await self._analyze_file_content_changes(
                    repo_path, commit_hash, file_path
                )
                functions_changed = content_analysis.get("functions", [])
                classes_changed = content_analysis.get("classes", [])
                imports_changed = content_analysis.get("imports", [])
            
            # Determine language and complexity
            language = self._detect_file_language(file_path)
            complexity_score = self._calculate_complexity_score(
                lines_added, lines_deleted, functions_changed, classes_changed
            )
            
            file_change = FileChange(
                file_path=file_path,
                change_type=change_type,
                lines_added=lines_added,
                lines_deleted=lines_deleted,
                functions_changed=functions_changed,
                classes_changed=classes_changed,
                imports_changed=imports_changed,
                language=language,
                complexity_score=complexity_score
            )
            
            file_changes.append(file_change)
        
        return file_changes
    
    async def _get_change_type(self, repo_path: str, commit_hash: str, file_path: str) -> str:
        """Determine the type of change for a file"""
        # Check if file was added, deleted, or renamed
        status_output = await self._run_git_command(
            repo_path,
            ["show", "--name-status", "--format=", commit_hash]
        )
        
        for line in status_output.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2 and parts[1] == file_path:
                status = parts[0][0]
                if status == 'A':
                    return "added"
                elif status == 'D':
                    return "deleted"
                elif status == 'R':
                    return "renamed"
                elif status == 'M':
                    return "modified"
        
        return "modified"  # Default
    
    async def _analyze_file_content_changes(
        self,
        repo_path: str,
        commit_hash: str,
        file_path: str
    ) -> Dict[str, List[str]]:
        """Analyze what functions/classes/imports changed in a file"""
        analysis = {"functions": [], "classes": [], "imports": []}
        
        # Get the diff for this file
        diff_output = await self._run_git_command(
            repo_path,
            ["show", commit_hash, "--", file_path]
        )
        
        if not diff_output:
            return analysis
        
        # Parse diff to find changed functions/classes
        language = self._detect_file_language(file_path)
        
        if language == "python":
            analysis = self._analyze_python_changes(diff_output)
        elif language == "javascript":
            analysis = self._analyze_javascript_changes(diff_output)
        # Add more language-specific analyzers as needed
        
        return analysis
    
    def _analyze_python_changes(self, diff_content: str) -> Dict[str, List[str]]:
        """Analyze Python file changes"""
        analysis = {"functions": [], "classes": [], "imports": []}
        
        # Find added/modified lines
        for line in diff_content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                clean_line = line[1:].strip()
                
                # Check for function definitions
                if clean_line.startswith('def '):
                    func_match = re.search(r'def\s+(\w+)', clean_line)
                    if func_match:
                        analysis["functions"].append(func_match.group(1))
                
                # Check for class definitions
                elif clean_line.startswith('class '):
                    class_match = re.search(r'class\s+(\w+)', clean_line)
                    if class_match:
                        analysis["classes"].append(class_match.group(1))
                
                # Check for imports
                elif clean_line.startswith(('import ', 'from ')):
                    analysis["imports"].append(clean_line)
        
        return analysis
    
    def _analyze_javascript_changes(self, diff_content: str) -> Dict[str, List[str]]:
        """Analyze JavaScript/TypeScript file changes"""
        analysis = {"functions": [], "classes": [], "imports": []}
        
        for line in diff_content.split('\n'):
            if line.startswith('+') and not line.startswith('+++'):
                clean_line = line[1:].strip()
                
                # Check for function definitions
                func_patterns = [
                    r'function\s+(\w+)',
                    r'const\s+(\w+)\s*=.*=>',
                    r'(\w+)\s*:\s*function',
                    r'(\w+)\s*\([^)]*\)\s*{'
                ]
                
                for pattern in func_patterns:
                    match = re.search(pattern, clean_line)
                    if match:
                        analysis["functions"].append(match.group(1))
                        break
                
                # Check for class definitions
                class_match = re.search(r'class\s+(\w+)', clean_line)
                if class_match:
                    analysis["classes"].append(class_match.group(1))
                
                # Check for imports
                if clean_line.startswith(('import ', 'const ', 'require(')):
                    analysis["imports"].append(clean_line)
        
        return analysis
    
    def _detect_file_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        for pattern, language in self.file_patterns.items():
            if re.search(pattern, file_path, re.IGNORECASE):
                return language
        return "unknown"
    
    def _calculate_complexity_score(
        self,
        lines_added: int,
        lines_deleted: int,
        functions_changed: List[str],
        classes_changed: List[str]
    ) -> int:
        """Calculate complexity score for file changes (1-10 scale)"""
        score = 1
        
        # Lines changed factor
        total_lines = lines_added + lines_deleted
        if total_lines > 200:
            score += 3
        elif total_lines > 100:
            score += 2
        elif total_lines > 50:
            score += 1
        
        # Functions/classes factor
        if len(functions_changed) > 10:
            score += 2
        elif len(functions_changed) > 5:
            score += 1
        
        if len(classes_changed) > 5:
            score += 2
        elif len(classes_changed) > 2:
            score += 1
        
        return min(score, 10)
    
    def _classify_commit_type(self, commit_message: str) -> CommitType:
        """Classify commit type based on message"""
        message_lower = commit_message.lower()
        
        for commit_type, patterns in self.commit_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return CommitType(commit_type)
        
        # Default classification based on keywords
        if any(word in message_lower for word in ["merge", "pull request"]):
            return CommitType.MERGE
        elif "revert" in message_lower:
            return CommitType.REVERT
        elif any(word in message_lower for word in ["release", "version"]):
            return CommitType.RELEASE
        elif any(word in message_lower for word in ["hotfix", "urgent", "critical"]):
            return CommitType.HOTFIX
        
        return CommitType.CHORE  # Default fallback
    
    def _assess_change_impact(self, file_changes: List[FileChange], commit_message: str) -> ChangeImpact:
        """Assess the impact level of changes"""
        # Check for breaking change indicators
        if any(keyword in commit_message.lower() for keyword in ["breaking", "major", "incompatible"]):
            return ChangeImpact.CRITICAL
        
        # Calculate impact based on changes
        total_lines = sum(fc.lines_added + fc.lines_deleted for fc in file_changes)
        total_files = len(file_changes)
        
        # High complexity or many files affected
        if total_lines > 500 or total_files > 20:
            return ChangeImpact.HIGH
        
        # Medium impact
        if total_lines > 100 or total_files > 5:
            return ChangeImpact.MEDIUM
        
        # Check file types for impact
        critical_files = sum(1 for fc in file_changes if any(
            pattern in fc.file_path.lower() for pattern in [
                "config", "security", "auth", "database", "migration"
            ]
        ))
        
        if critical_files > 0:
            return ChangeImpact.HIGH
        
        # Documentation or test only changes
        if all(fc.language in ["documentation", "test"] for fc in file_changes):
            return ChangeImpact.LOW
        
        return ChangeImpact.MEDIUM  # Default
    
    def _extract_semantic_tags(self, commit_message: str) -> List[str]:
        """Extract semantic tags from commit message"""
        tags = []
        message_lower = commit_message.lower()
        
        for category, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    tags.append(category)
                    break
        
        return tags
    
    def _extract_issue_references(self, commit_message: str) -> List[str]:
        """Extract issue/ticket references from commit message"""
        # Common patterns for issue references
        patterns = [
            r"#(\d+)",           # GitHub issues
            r"fixes?\s+#(\d+)",  # Fixes #123
            r"closes?\s+#(\d+)", # Closes #123
            r"JIRA-(\d+)",       # JIRA tickets
            r"([A-Z]+-\d+)",     # Generic ticket format
        ]
        
        issues = []
        for pattern in patterns:
            matches = re.findall(pattern, commit_message, re.IGNORECASE)
            issues.extend(matches)
        
        return list(set(issues))  # Remove duplicates
    
    def _extract_breaking_changes(self, commit_message: str) -> List[str]:
        """Extract breaking change descriptions"""
        breaking_changes = []
        
        # Look for BREAKING CHANGE: section
        breaking_match = re.search(
            r"BREAKING CHANGE:\s*(.*?)(?:\n\n|\n[A-Z]|$)",
            commit_message,
            re.DOTALL | re.IGNORECASE
        )
        
        if breaking_match:
            breaking_changes.append(breaking_match.group(1).strip())
        
        return breaking_changes
    
    async def _get_current_branch(self, repo_path: str) -> str:
        """Get current branch name"""
        output = await self._run_git_command(repo_path, ["branch", "--show-current"])
        return output.strip()
    
    async def _store_commit_context(self, context: CommitContext):
        """Store commit context to disk"""
        try:
            commit_file = self.commits_dir / f"{context.commit_hash}.json"
            with open(commit_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(context), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to store commit context: {e}")
    
    async def _integrate_with_memory(self, context: CommitContext):
        """Integrate commit context with memory system"""
        try:
            # Create memory entry for commit
            memory_content = self._format_commit_memory(context)
            
            await self.memory_system.add_memory(
                content=memory_content,
                memory_type="development",
                priority="medium",
                tags=[
                    "git_commit",
                    f"type:{context.commit_type.value}",
                    f"impact:{context.change_impact.value}",
                    f"branch:{context.branch_name}",
                    f"author:{context.author}"
                ] + context.semantic_tags,
                metadata={
                    "commit_hash": context.commit_hash,
                    "commit_type": context.commit_type.value,
                    "change_impact": context.change_impact.value,
                    "files_count": len(context.files_changed),
                    "lines_changed": context.total_lines_added + context.total_lines_deleted,
                    "branch": context.branch_name,
                    "timestamp": context.timestamp
                }
            )
            
            logger.debug(f"Integrated commit {context.commit_hash[:12]} with memory system")
            
        except Exception as e:
            logger.error(f"Failed to integrate with memory system: {e}")
    
    def _format_commit_memory(self, context: CommitContext) -> str:
        """Format commit context for memory storage"""
        content = f"""Git Commit: {context.commit_hash[:12]}
Author: {context.author}
Date: {context.timestamp}
Branch: {context.branch_name}
Type: {context.commit_type.value}
Impact: {context.change_impact.value}

Message: {context.commit_message}

Changes:
- Files changed: {len(context.files_changed)}
- Lines added: {context.total_lines_added}
- Lines deleted: {context.total_lines_deleted}

Key files modified:"""
        
        # Add most significant file changes
        significant_files = sorted(
            context.files_changed,
            key=lambda fc: fc.lines_added + fc.lines_deleted,
            reverse=True
        )[:5]
        
        for file_change in significant_files:
            content += f"\n- {file_change.file_path} (+{file_change.lines_added}/-{file_change.lines_deleted})"
            if file_change.functions_changed:
                content += f" [functions: {', '.join(file_change.functions_changed[:3])}]"
        
        if context.semantic_tags:
            content += f"\n\nSemantic tags: {', '.join(context.semantic_tags)}"
        
        if context.related_issues:
            content += f"\nRelated issues: {', '.join(context.related_issues)}"
        
        if context.breaking_changes:
            content += f"\nBreaking changes: {'; '.join(context.breaking_changes)}"
        
        return content
    
    async def capture_branch_context(self, repository_path: str, branch_name: str) -> Optional[BranchContext]:
        """Capture context about a git branch"""
        try:
            # Get branch information
            branch_info = await self._get_branch_info(repository_path, branch_name)
            if not branch_info:
                return None
            
            # Create branch context
            context = BranchContext(
                branch_name=branch_name,
                base_branch=branch_info["base_branch"],
                created_date=branch_info["created_date"],
                last_activity=branch_info["last_activity"],
                commit_count=branch_info["commit_count"],
                total_files_changed=branch_info["files_changed"],
                total_lines_changed=branch_info["lines_changed"],
                contributors=branch_info["contributors"],
                feature_description=branch_info.get("description", ""),
                related_issues=self._extract_branch_issues(branch_name)
            )
            
            # Store branch context
            await self._store_branch_context(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to capture branch context: {e}")
            return None
    
    async def _get_branch_info(self, repo_path: str, branch_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a branch"""
        # This would implement detailed branch analysis
        # For now, return a simplified version
        return {
            "base_branch": "main",
            "created_date": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "commit_count": 5,
            "files_changed": 10,
            "lines_changed": 200,
            "contributors": ["author1", "author2"],
            "description": f"Development branch: {branch_name}"
        }
    
    def _extract_branch_issues(self, branch_name: str) -> List[str]:
        """Extract issue references from branch name"""
        # Common branch naming patterns with issue references
        patterns = [
            r"feature[/-](\d+)",
            r"bug[/-](\d+)",
            r"issue[/-](\d+)",
            r"([A-Z]+-\d+)"
        ]
        
        issues = []
        for pattern in patterns:
            matches = re.findall(pattern, branch_name, re.IGNORECASE)
            issues.extend(matches)
        
        return issues
    
    async def _store_branch_context(self, context: BranchContext):
        """Store branch context to disk"""
        try:
            branch_file = self.branches_dir / f"{context.branch_name.replace('/', '_')}.json"
            with open(branch_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(context), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to store branch context: {e}")
    
    async def setup_git_hooks(self, repository_path: str) -> bool:
        """Setup git hooks for automatic commit capture"""
        try:
            hooks_dir = Path(repository_path) / ".git" / "hooks"
            
            # Post-commit hook
            post_commit_hook = hooks_dir / "post-commit"
            hook_content = f"""#!/bin/bash
# Auto-generated git hook for commit context capture
python3 -c "
import sys
sys.path.append('{MEMORY_SYSTEM_PATH}')
from git_commit_capture import GitCommitCapture
import asyncio

async def capture():
    capture = GitCommitCapture()
    await capture.capture_commit_context('{repository_path}')

asyncio.run(capture())
"
"""
            
            with open(post_commit_hook, 'w') as f:
                f.write(hook_content)
            
            # Make executable
            os.chmod(post_commit_hook, 0o755)
            
            logger.info(f"Git hooks installed in {repository_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup git hooks: {e}")
            return False
    
    async def get_repository_activity_summary(
        self,
        repository_path: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get repository activity summary"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Get commit statistics
            commit_count_output = await self._run_git_command(
                repository_path,
                ["rev-list", "--count", "--since=" + since_date, "HEAD"]
            )
            
            commit_count = int(commit_count_output.strip()) if commit_count_output.strip() else 0
            
            # Get contributor statistics
            contributors_output = await self._run_git_command(
                repository_path,
                ["shortlog", "-sn", "--since=" + since_date, "HEAD"]
            )
            
            contributors = []
            for line in contributors_output.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        contributors.append({
                            "name": parts[1],
                            "commits": int(parts[0])
                        })
            
            # Get file change statistics
            files_output = await self._run_git_command(
                repository_path,
                ["diff", "--name-only", f"HEAD~{commit_count}", "HEAD"]
            )
            
            changed_files = [f for f in files_output.strip().split('\n') if f.strip()]
            
            return {
                "period_days": days,
                "total_commits": commit_count,
                "contributors": contributors,
                "files_changed": len(changed_files),
                "most_active_files": changed_files[:10],
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get repository activity summary: {e}")
            return {}


# Global git capture instance
git_capture = GitCommitCapture()

# Convenience functions
async def capture_commit(repository_path: str, commit_hash: Optional[str] = None) -> Optional[CommitContext]:
    """Capture commit context"""
    return await git_capture.capture_commit_context(repository_path, commit_hash)

async def capture_branch(repository_path: str, branch_name: str) -> Optional[BranchContext]:
    """Capture branch context"""
    return await git_capture.capture_branch_context(repository_path, branch_name)

async def setup_auto_capture(repository_path: str) -> bool:
    """Setup automatic commit capture"""
    return await git_capture.setup_git_hooks(repository_path)

async def get_repo_activity(repository_path: str, days: int = 30) -> Dict[str, Any]:
    """Get repository activity summary"""
    return await git_capture.get_repository_activity_summary(repository_path, days)

if __name__ == "__main__":
    # Test git commit capture
    async def test_git_capture():
        print("📝 Testing Git Commit Context Capture")
        
        # Test with current repository
        repo_path = "/opt/projects/knowledgehub"
        
        if os.path.exists(repo_path):
            print(f"Testing with repository: {repo_path}")
            
            # Capture latest commit
            context = await capture_commit(repo_path)
            if context:
                print(f"✅ Captured commit: {context.commit_hash[:12]}")
                print(f"   Type: {context.commit_type.value}")
                print(f"   Impact: {context.change_impact.value}")
                print(f"   Files changed: {len(context.files_changed)}")
                print(f"   Lines changed: {context.total_lines_added + context.total_lines_deleted}")
            
            # Get activity summary
            activity = await get_repo_activity(repo_path, 7)
            if activity:
                print(f"✅ Repository activity (7 days):")
                print(f"   Commits: {activity.get('total_commits', 0)}")
                print(f"   Contributors: {len(activity.get('contributors', []))}")
                print(f"   Files changed: {activity.get('files_changed', 0)}")
        
        print("✅ Git Commit Capture ready!")
    
    asyncio.run(test_git_capture())