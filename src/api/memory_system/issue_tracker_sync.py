#!/usr/bin/env python3
"""
Issue Tracker Synchronization System
Synchronizes with various issue tracking systems to capture development context and project insights
"""

import os
import sys
import json
import asyncio
import logging
import aiohttp
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
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

class IssueStatus(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    TESTING = "testing"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    BACKLOG = "backlog"

class IssuePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"

class IssueType(Enum):
    BUG = "bug"
    FEATURE = "feature"
    ENHANCEMENT = "enhancement"
    TASK = "task"
    STORY = "story"
    EPIC = "epic"
    SUBTASK = "subtask"
    QUESTION = "question"
    DOCUMENTATION = "documentation"

class TrackerProvider(Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    JIRA = "jira"
    AZURE_DEVOPS = "azure_devops"
    LINEAR = "linear"
    TRELLO = "trello"
    ASANA = "asana"
    CLICKUP = "clickup"
    GENERIC = "generic"

@dataclass
class IssueComment:
    """Issue comment/update"""
    comment_id: str
    author: str
    author_email: str = ""
    content: str = ""
    created_at: str = ""
    updated_at: str = ""
    
    # Comment metadata
    is_system_comment: bool = False
    comment_type: str = "comment"  # comment, status_change, assignment, etc.
    
    # Extracted insights
    mentioned_users: List[str] = None
    referenced_issues: List[str] = None
    code_references: List[str] = None
    
    def __post_init__(self):
        if self.mentioned_users is None:
            self.mentioned_users = []
        if self.referenced_issues is None:
            self.referenced_issues = []
        if self.code_references is None:
            self.code_references = []

@dataclass
class IssueLabel:
    """Issue label/tag"""
    name: str
    color: str = ""
    description: str = ""
    category: str = ""  # bug, feature, priority, etc.

@dataclass
class IssueRelation:
    """Relationship between issues"""
    related_issue_id: str
    relation_type: str  # blocks, blocked_by, relates_to, duplicate_of, etc.
    description: str = ""

@dataclass
class WorklogEntry:
    """Time tracking entry"""
    entry_id: str
    author: str
    time_spent_minutes: int
    description: str = ""
    date: str = ""
    
    # Activity classification
    activity_type: str = "development"  # development, testing, review, documentation
    billable: bool = True

@dataclass
class Issue:
    """Complete issue/ticket information"""
    issue_id: str
    title: str
    description: str
    status: IssueStatus
    priority: IssuePriority
    issue_type: IssueType
    
    # People
    assignee: str = ""
    reporter: str = ""
    
    # Tracking
    created_at: str = ""
    updated_at: str = ""
    resolved_at: str = ""
    
    # Organization
    project: str = ""
    milestone: str = ""
    sprint: str = ""
    epic: str = ""
    
    # Metadata
    labels: List[IssueLabel] = None
    comments: List[IssueComment] = None
    relations: List[IssueRelation] = None
    worklog: List[WorklogEntry] = None
    
    # Development context
    linked_commits: List[str] = None
    linked_pull_requests: List[str] = None
    linked_branches: List[str] = None
    
    # Estimates and tracking
    story_points: Optional[int] = None
    original_estimate_minutes: Optional[int] = None
    remaining_estimate_minutes: Optional[int] = None
    time_spent_minutes: int = 0
    
    # Provider specific
    provider: TrackerProvider = TrackerProvider.GENERIC
    provider_url: str = ""
    external_id: str = ""
    
    # Memory integration
    memory_id: str = ""
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = []
        if self.comments is None:
            self.comments = []
        if self.relations is None:
            self.relations = []
        if self.worklog is None:
            self.worklog = []
        if self.linked_commits is None:
            self.linked_commits = []
        if self.linked_pull_requests is None:
            self.linked_pull_requests = []
        if self.linked_branches is None:
            self.linked_branches = []
        if not self.memory_id:
            self.memory_id = f"issue_{self.issue_id}"

@dataclass
class ProjectMetrics:
    """Project-level metrics from issue tracking"""
    project_name: str
    total_issues: int
    open_issues: int
    closed_issues: int
    
    # Status breakdown
    status_breakdown: Dict[str, int] = None
    
    # Priority breakdown
    priority_breakdown: Dict[str, int] = None
    
    # Type breakdown
    type_breakdown: Dict[str, int] = None
    
    # Time metrics
    avg_resolution_time_hours: float = 0.0
    avg_response_time_hours: float = 0.0
    
    # Velocity
    issues_closed_last_week: int = 0
    issues_created_last_week: int = 0
    story_points_completed: int = 0
    
    # Team metrics
    active_contributors: List[str] = None
    top_assignees: Dict[str, int] = None
    
    # Generated timestamp
    generated_at: str = ""
    
    def __post_init__(self):
        if self.status_breakdown is None:
            self.status_breakdown = {}
        if self.priority_breakdown is None:
            self.priority_breakdown = {}
        if self.type_breakdown is None:
            self.type_breakdown = {}
        if self.active_contributors is None:
            self.active_contributors = []
        if self.top_assignees is None:
            self.top_assignees = {}
        if not self.generated_at:
            self.generated_at = datetime.now().isoformat()

class IssueTrackerSync:
    """
    Synchronizes with issue tracking systems for development workflow integration
    """
    
    def __init__(self, memory_system: Optional[UnifiedMemorySystem] = None):
        self.memory_system = memory_system or UnifiedMemorySystem()
        
        # Storage
        self.issues_dir = Path("/opt/projects/memory-system/data/issue_tracker")
        self.projects_dir = Path("/opt/projects/memory-system/data/project_metrics")
        self.sync_logs_dir = Path("/opt/projects/memory-system/data/sync_logs")
        
        for directory in [self.issues_dir, self.projects_dir, self.sync_logs_dir]:
            directory.mkdir(exist_ok=True)
        
        # Provider configurations
        self.providers = {}
        
        # Configuration
        self.config = {
            "auto_sync_enabled": True,
            "sync_interval_minutes": 15,
            "full_sync_interval_hours": 24,
            "memory_integration": True,
            "track_comments": True,
            "track_worklog": True,
            "retention_days": 365
        }
        
        # Cached data
        self.issue_cache: Dict[str, Issue] = {}
        self.project_metrics: Dict[str, ProjectMetrics] = {}
        
        # Load existing data
        self._load_recent_issues()
    
    def _load_recent_issues(self):
        """Load recent issues from storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for issue_file in self.issues_dir.glob("*.json"):
                try:
                    with open(issue_file, 'r', encoding='utf-8') as f:
                        issue_data = json.load(f)
                    
                    # Check if recent enough
                    updated_at = datetime.fromisoformat(issue_data["updated_at"])
                    if updated_at >= cutoff_date:
                        issue = Issue(**issue_data)
                        self.issue_cache[issue.issue_id] = issue
                        
                except Exception as e:
                    logger.warning(f"Failed to load issue file {issue_file}: {e}")
            
            logger.info(f"Loaded {len(self.issue_cache)} recent issues")
            
        except Exception as e:
            logger.error(f"Failed to load recent issues: {e}")
    
    async def setup_github_integration(
        self,
        github_token: str,
        repository: str
    ) -> bool:
        """Setup GitHub Issues integration"""
        try:
            self.providers[TrackerProvider.GITHUB] = {
                "token": github_token,
                "repository": repository,
                "api_base": "https://api.github.com"
            }
            
            # Test connection
            headers = {"Authorization": f"token {github_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.github.com/repos/{repository}/issues?state=all&per_page=1",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"GitHub Issues integration setup for {repository}")
                        return True
                    else:
                        logger.error(f"GitHub API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to setup GitHub integration: {e}")
            return False
    
    async def setup_jira_integration(
        self,
        jira_url: str,
        username: str,
        api_token: str,
        project_key: str
    ) -> bool:
        """Setup JIRA integration"""
        try:
            self.providers[TrackerProvider.JIRA] = {
                "url": jira_url,
                "username": username,
                "token": api_token,
                "project_key": project_key
            }
            
            # Test connection
            auth = aiohttp.BasicAuth(username, api_token)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{jira_url}/rest/api/2/project/{project_key}",
                    auth=auth
                ) as response:
                    if response.status == 200:
                        logger.info(f"JIRA integration setup for project {project_key}")
                        return True
                    else:
                        logger.error(f"JIRA API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to setup JIRA integration: {e}")
            return False
    
    async def setup_gitlab_integration(
        self,
        gitlab_token: str,
        project_id: str,
        gitlab_url: str = "https://gitlab.com"
    ) -> bool:
        """Setup GitLab Issues integration"""
        try:
            self.providers[TrackerProvider.GITLAB] = {
                "token": gitlab_token,
                "project_id": project_id,
                "api_base": f"{gitlab_url}/api/v4"
            }
            
            # Test connection
            headers = {"PRIVATE-TOKEN": gitlab_token}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{gitlab_url}/api/v4/projects/{project_id}/issues?per_page=1",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"GitLab Issues integration setup for project {project_id}")
                        return True
                    else:
                        logger.error(f"GitLab API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to setup GitLab integration: {e}")
            return False
    
    async def sync_github_issues(
        self,
        repository: str,
        full_sync: bool = False
    ) -> int:
        """Sync issues from GitHub"""
        try:
            provider_config = self.providers.get(TrackerProvider.GITHUB)
            if not provider_config:
                logger.error("GitHub not configured")
                return 0
            
            headers = {"Authorization": f"token {provider_config['token']}"}
            
            # Determine sync strategy
            since_param = ""
            if not full_sync:
                since_date = (datetime.now() - timedelta(hours=1)).isoformat() + "Z"
                since_param = f"&since={since_date}"
            
            synced_count = 0
            page = 1
            per_page = 100
            
            async with aiohttp.ClientSession() as session:
                while True:
                    # Get issues
                    async with session.get(
                        f"https://api.github.com/repos/{repository}/issues"
                        f"?state=all&per_page={per_page}&page={page}{since_param}",
                        headers=headers
                    ) as response:
                        if response.status != 200:
                            logger.error(f"GitHub API error: {response.status}")
                            break
                        
                        issues_data = await response.json()
                        if not issues_data:
                            break
                        
                        # Process each issue
                        for issue_data in issues_data:
                            # Skip pull requests (GitHub treats them as issues)
                            if issue_data.get("pull_request"):
                                continue
                            
                            issue = await self._parse_github_issue(issue_data, repository, headers, session)
                            if issue:
                                await self._store_issue(issue)
                                synced_count += 1
                        
                        page += 1
                        
                        # Break if this is the last page
                        if len(issues_data) < per_page:
                            break
            
            logger.info(f"Synced {synced_count} issues from GitHub")
            return synced_count
            
        except Exception as e:
            logger.error(f"Failed to sync GitHub issues: {e}")
            return 0
    
    async def _parse_github_issue(
        self,
        issue_data: Dict[str, Any],
        repository: str,
        headers: Dict[str, str],
        session: aiohttp.ClientSession
    ) -> Optional[Issue]:
        """Parse GitHub issue data"""
        try:
            # Basic issue info
            issue_id = str(issue_data["number"])
            title = issue_data["title"]
            description = issue_data.get("body") or ""
            
            # Status mapping
            status = IssueStatus.OPEN if issue_data["state"] == "open" else IssueStatus.CLOSED
            
            # Priority from labels (GitHub doesn't have native priority)
            priority = IssuePriority.MEDIUM  # Default
            issue_type = IssueType.BUG  # Default
            
            labels = []
            for label_data in issue_data.get("labels", []):
                label = IssueLabel(
                    name=label_data["name"],
                    color=label_data["color"],
                    description=label_data.get("description") or ""
                )
                labels.append(label)
                
                # Extract priority and type from labels
                label_name = label_data["name"].lower()
                if "critical" in label_name or "urgent" in label_name:
                    priority = IssuePriority.CRITICAL
                elif "high" in label_name:
                    priority = IssuePriority.HIGH
                elif "low" in label_name:
                    priority = IssuePriority.LOW
                
                if "feature" in label_name or "enhancement" in label_name:
                    issue_type = IssueType.FEATURE
                elif "documentation" in label_name or "docs" in label_name:
                    issue_type = IssueType.DOCUMENTATION
                elif "question" in label_name:
                    issue_type = IssueType.QUESTION
            
            # Get comments
            comments = []
            if self.config["track_comments"]:
                comments = await self._get_github_comments(
                    repository, issue_id, headers, session
                )
            
            # Extract development links
            linked_commits = self._extract_commit_references(description + " " + " ".join(c.content for c in comments))
            linked_prs = self._extract_pr_references(description + " " + " ".join(c.content for c in comments))
            
            return Issue(
                issue_id=issue_id,
                title=title,
                description=description,
                status=status,
                priority=priority,
                issue_type=issue_type,
                assignee=issue_data.get("assignee", {}).get("login", "") if issue_data.get("assignee") else "",
                reporter=issue_data["user"]["login"],
                created_at=issue_data["created_at"],
                updated_at=issue_data["updated_at"],
                resolved_at=issue_data.get("closed_at", ""),
                project=repository,
                milestone=issue_data.get("milestone", {}).get("title", "") if issue_data.get("milestone") else "",
                labels=labels,
                comments=comments,
                linked_commits=linked_commits,
                linked_pull_requests=linked_prs,
                provider=TrackerProvider.GITHUB,
                provider_url=issue_data["html_url"],
                external_id=str(issue_data["id"])
            )
            
        except Exception as e:
            logger.error(f"Failed to parse GitHub issue: {e}")
            return None
    
    async def _get_github_comments(
        self,
        repository: str,
        issue_number: str,
        headers: Dict[str, str],
        session: aiohttp.ClientSession
    ) -> List[IssueComment]:
        """Get comments for a GitHub issue"""
        comments = []
        
        try:
            async with session.get(
                f"https://api.github.com/repos/{repository}/issues/{issue_number}/comments",
                headers=headers
            ) as response:
                if response.status == 200:
                    comments_data = await response.json()
                    
                    for comment_data in comments_data:
                        comment = IssueComment(
                            comment_id=str(comment_data["id"]),
                            author=comment_data["user"]["login"],
                            content=comment_data.get("body") or "",
                            created_at=comment_data["created_at"],
                            updated_at=comment_data["updated_at"],
                            mentioned_users=self._extract_user_mentions(comment_data.get("body") or ""),
                            referenced_issues=self._extract_issue_references(comment_data.get("body") or ""),
                            code_references=self._extract_code_references(comment_data.get("body") or "")
                        )
                        comments.append(comment)
        
        except Exception as e:
            logger.error(f"Failed to get GitHub comments: {e}")
        
        return comments
    
    async def sync_jira_issues(
        self,
        project_key: str,
        full_sync: bool = False
    ) -> int:
        """Sync issues from JIRA"""
        try:
            provider_config = self.providers.get(TrackerProvider.JIRA)
            if not provider_config:
                logger.error("JIRA not configured")
                return 0
            
            auth = aiohttp.BasicAuth(provider_config["username"], provider_config["token"])
            
            # Build JQL query
            jql = f"project = {project_key}"
            if not full_sync:
                since_date = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
                jql += f" AND updated >= '{since_date}'"
            
            synced_count = 0
            start_at = 0
            max_results = 50
            
            async with aiohttp.ClientSession() as session:
                while True:
                    # Get issues
                    url = f"{provider_config['url']}/rest/api/2/search"
                    params = {
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": max_results,
                        "expand": "changelog"
                    }
                    
                    async with session.get(url, auth=auth, params=params) as response:
                        if response.status != 200:
                            logger.error(f"JIRA API error: {response.status}")
                            break
                        
                        search_result = await response.json()
                        issues_data = search_result.get("issues", [])
                        
                        if not issues_data:
                            break
                        
                        # Process each issue
                        for issue_data in issues_data:
                            issue = await self._parse_jira_issue(issue_data, provider_config, session)
                            if issue:
                                await self._store_issue(issue)
                                synced_count += 1
                        
                        start_at += max_results
                        
                        # Break if we've processed all issues
                        if start_at >= search_result.get("total", 0):
                            break
            
            logger.info(f"Synced {synced_count} issues from JIRA")
            return synced_count
            
        except Exception as e:
            logger.error(f"Failed to sync JIRA issues: {e}")
            return 0
    
    async def _parse_jira_issue(
        self,
        issue_data: Dict[str, Any],
        provider_config: Dict[str, str],
        session: aiohttp.ClientSession
    ) -> Optional[Issue]:
        """Parse JIRA issue data"""
        try:
            fields = issue_data["fields"]
            
            # Basic issue info
            issue_id = issue_data["key"]
            title = fields["summary"]
            description = fields.get("description") or ""
            
            # Status mapping
            status_name = fields["status"]["name"].lower()
            status = self._map_jira_status(status_name)
            
            # Priority mapping
            priority_name = fields.get("priority", {}).get("name", "Medium") if fields.get("priority") else "Medium"
            priority = self._map_jira_priority(priority_name)
            
            # Issue type mapping
            type_name = fields["issuetype"]["name"].lower()
            issue_type = self._map_jira_issue_type(type_name)
            
            # Get worklog if enabled
            worklog = []
            if self.config["track_worklog"]:
                worklog = await self._get_jira_worklog(issue_data["key"], provider_config, session)
            
            return Issue(
                issue_id=issue_id,
                title=title,
                description=description,
                status=status,
                priority=priority,
                issue_type=issue_type,
                assignee=fields.get("assignee", {}).get("displayName", "") if fields.get("assignee") else "",
                reporter=fields.get("reporter", {}).get("displayName", "") if fields.get("reporter") else "",
                created_at=fields["created"],
                updated_at=fields["updated"],
                resolved_at=fields.get("resolutiondate", ""),
                project=fields["project"]["key"],
                story_points=fields.get("customfield_10002"),  # Common story points field
                original_estimate_minutes=self._convert_jira_time(fields.get("timeoriginalestimate")),
                remaining_estimate_minutes=self._convert_jira_time(fields.get("timeestimate")),
                time_spent_minutes=self._convert_jira_time(fields.get("timespent")),
                worklog=worklog,
                provider=TrackerProvider.JIRA,
                provider_url=f"{provider_config['url']}/browse/{issue_data['key']}",
                external_id=issue_data["id"]
            )
            
        except Exception as e:
            logger.error(f"Failed to parse JIRA issue: {e}")
            return None
    
    def _map_jira_status(self, status_name: str) -> IssueStatus:
        """Map JIRA status to standard status"""
        status_map = {
            "to do": IssueStatus.BACKLOG,
            "backlog": IssueStatus.BACKLOG,
            "open": IssueStatus.OPEN,
            "in progress": IssueStatus.IN_PROGRESS,
            "code review": IssueStatus.REVIEW,
            "testing": IssueStatus.TESTING,
            "done": IssueStatus.CLOSED,
            "closed": IssueStatus.CLOSED,
            "cancelled": IssueStatus.CANCELLED,
            "blocked": IssueStatus.BLOCKED
        }
        
        return status_map.get(status_name, IssueStatus.OPEN)
    
    def _map_jira_priority(self, priority_name: str) -> IssuePriority:
        """Map JIRA priority to standard priority"""
        priority_map = {
            "highest": IssuePriority.CRITICAL,
            "high": IssuePriority.HIGH,
            "medium": IssuePriority.MEDIUM,
            "low": IssuePriority.LOW,
            "lowest": IssuePriority.TRIVIAL
        }
        
        return priority_map.get(priority_name.lower(), IssuePriority.MEDIUM)
    
    def _map_jira_issue_type(self, type_name: str) -> IssueType:
        """Map JIRA issue type to standard type"""
        type_map = {
            "bug": IssueType.BUG,
            "story": IssueType.STORY,
            "task": IssueType.TASK,
            "epic": IssueType.EPIC,
            "sub-task": IssueType.SUBTASK,
            "improvement": IssueType.ENHANCEMENT,
            "new feature": IssueType.FEATURE
        }
        
        return type_map.get(type_name, IssueType.TASK)
    
    def _convert_jira_time(self, time_value: Optional[int]) -> Optional[int]:
        """Convert JIRA time (seconds) to minutes"""
        return int(time_value / 60) if time_value else None
    
    async def _get_jira_worklog(
        self,
        issue_key: str,
        provider_config: Dict[str, str],
        session: aiohttp.ClientSession
    ) -> List[WorklogEntry]:
        """Get worklog entries for a JIRA issue"""
        worklog = []
        
        try:
            auth = aiohttp.BasicAuth(provider_config["username"], provider_config["token"])
            
            async with session.get(
                f"{provider_config['url']}/rest/api/2/issue/{issue_key}/worklog",
                auth=auth
            ) as response:
                if response.status == 200:
                    worklog_data = await response.json()
                    
                    for entry_data in worklog_data.get("worklogs", []):
                        entry = WorklogEntry(
                            entry_id=entry_data["id"],
                            author=entry_data["author"]["displayName"],
                            time_spent_minutes=int(entry_data["timeSpentSeconds"] / 60),
                            description=entry_data.get("comment", ""),
                            date=entry_data["started"]
                        )
                        worklog.append(entry)
        
        except Exception as e:
            logger.error(f"Failed to get JIRA worklog: {e}")
        
        return worklog
    
    def _extract_user_mentions(self, text: str) -> List[str]:
        """Extract user mentions from text"""
        mentions = re.findall(r'@([a-zA-Z0-9_-]+)', text)
        return list(set(mentions))
    
    def _extract_issue_references(self, text: str) -> List[str]:
        """Extract issue references from text"""
        # Common patterns for issue references
        patterns = [
            r'#(\d+)',           # GitHub style
            r'([A-Z]+-\d+)',     # JIRA style
            r'issue[s]?\s+(\d+)', # "issue 123"
            r'fixes?\s+#(\d+)',  # "fixes #123"
            r'closes?\s+#(\d+)'  # "closes #123"
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))
    
    def _extract_commit_references(self, text: str) -> List[str]:
        """Extract commit hash references from text"""
        # SHA patterns
        sha_pattern = r'\b([a-f0-9]{7,40})\b'
        commits = re.findall(sha_pattern, text, re.IGNORECASE)
        return [commit for commit in commits if len(commit) >= 7]
    
    def _extract_pr_references(self, text: str) -> List[str]:
        """Extract pull request references from text"""
        pr_patterns = [
            r'(?:pr|pull request)[s]?\s+#?(\d+)',
            r'merge request[s]?\s+!(\d+)',  # GitLab style
            r'#(\d+)'  # Generic number reference
        ]
        
        prs = []
        for pattern in pr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            prs.extend(matches)
        
        return list(set(prs))
    
    def _extract_code_references(self, text: str) -> List[str]:
        """Extract code references from text"""
        # Look for file paths, function names, etc.
        code_patterns = [
            r'`([^`]+)`',  # Inline code
            r'```[^`]*```',  # Code blocks
            r'(\w+\.py:\d+)',  # File:line references
            r'(\w+\.\w+\(\))',  # Function calls
        ]
        
        references = []
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            references.extend(matches)
        
        return references
    
    async def _store_issue(self, issue: Issue):
        """Store issue to disk and cache"""
        try:
            # Store to file
            issue_file = self.issues_dir / f"{issue.provider.value}_{issue.issue_id}.json"
            with open(issue_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(issue), f, indent=2, default=str)
            
            # Update cache
            self.issue_cache[issue.issue_id] = issue
            
            # Integrate with memory if enabled
            if self.config["memory_integration"]:
                await self._integrate_issue_with_memory(issue)
            
        except Exception as e:
            logger.error(f"Failed to store issue {issue.issue_id}: {e}")
    
    async def _integrate_issue_with_memory(self, issue: Issue):
        """Integrate issue with memory system"""
        try:
            # Create memory entry for issue
            memory_content = self._format_issue_memory(issue)
            
            tags = [
                "issue_tracker",
                f"provider:{issue.provider.value}",
                f"status:{issue.status.value}",
                f"priority:{issue.priority.value}",
                f"type:{issue.issue_type.value}",
                f"project:{issue.project}"
            ]
            
            # Add label-based tags
            tags.extend([f"label:{label.name}" for label in issue.labels[:5]])
            
            await self.memory_system.add_memory(
                content=memory_content,
                memory_type="development",
                priority="medium",
                tags=tags,
                metadata={
                    "issue_id": issue.issue_id,
                    "provider": issue.provider.value,
                    "status": issue.status.value,
                    "priority": issue.priority.value,
                    "issue_type": issue.issue_type.value,
                    "assignee": issue.assignee,
                    "project": issue.project,
                    "story_points": issue.story_points,
                    "time_spent_minutes": issue.time_spent_minutes,
                    "comments_count": len(issue.comments),
                    "linked_commits": len(issue.linked_commits),
                    "created_at": issue.created_at,
                    "updated_at": issue.updated_at
                }
            )
            
            logger.debug(f"Integrated issue {issue.issue_id} with memory system")
            
        except Exception as e:
            logger.error(f"Failed to integrate issue with memory: {e}")
    
    def _format_issue_memory(self, issue: Issue) -> str:
        """Format issue for memory storage"""
        content = f"""Issue: {issue.issue_id} - {issue.title}
Provider: {issue.provider.value}
Status: {issue.status.value}
Priority: {issue.priority.value}
Type: {issue.issue_type.value}
Assignee: {issue.assignee or 'Unassigned'}
Reporter: {issue.reporter}
Project: {issue.project}

Description:
{issue.description[:500]}{"..." if len(issue.description) > 500 else ""}"""
        
        if issue.story_points:
            content += f"\nStory Points: {issue.story_points}"
        
        if issue.time_spent_minutes:
            hours = issue.time_spent_minutes / 60
            content += f"\nTime Spent: {hours:.1f} hours"
        
        if issue.labels:
            label_names = [label.name for label in issue.labels[:5]]
            content += f"\nLabels: {', '.join(label_names)}"
        
        if issue.linked_commits:
            content += f"\nLinked Commits: {len(issue.linked_commits)}"
        
        if issue.linked_pull_requests:
            content += f"\nLinked PRs: {len(issue.linked_pull_requests)}"
        
        if issue.comments:
            content += f"\nComments: {len(issue.comments)}"
            # Add recent comments
            recent_comments = sorted(issue.comments, key=lambda c: c.created_at, reverse=True)[:2]
            for comment in recent_comments:
                content += f"\n  - {comment.author}: {comment.content[:100]}..."
        
        return content
    
    async def generate_project_metrics(self, project_name: str) -> ProjectMetrics:
        """Generate project metrics from synced issues"""
        try:
            # Filter issues for this project
            project_issues = [
                issue for issue in self.issue_cache.values()
                if issue.project == project_name
            ]
            
            if not project_issues:
                return ProjectMetrics(
                    project_name=project_name,
                    total_issues=0,
                    open_issues=0,
                    closed_issues=0
                )
            
            # Basic counts
            total_issues = len(project_issues)
            open_issues = len([i for i in project_issues if i.status in [IssueStatus.OPEN, IssueStatus.IN_PROGRESS, IssueStatus.REVIEW]])
            closed_issues = len([i for i in project_issues if i.status == IssueStatus.CLOSED])
            
            # Status breakdown
            status_breakdown = defaultdict(int)
            for issue in project_issues:
                status_breakdown[issue.status.value] += 1
            
            # Priority breakdown
            priority_breakdown = defaultdict(int)
            for issue in project_issues:
                priority_breakdown[issue.priority.value] += 1
            
            # Type breakdown
            type_breakdown = defaultdict(int)
            for issue in project_issues:
                type_breakdown[issue.issue_type.value] += 1
            
            # Time metrics
            closed_with_times = [
                issue for issue in project_issues
                if issue.status == IssueStatus.CLOSED and issue.resolved_at and issue.created_at
            ]
            
            avg_resolution_time = 0.0
            if closed_with_times:
                resolution_times = []
                for issue in closed_with_times:
                    created = datetime.fromisoformat(issue.created_at.replace('Z', '+00:00'))
                    resolved = datetime.fromisoformat(issue.resolved_at.replace('Z', '+00:00'))
                    resolution_times.append((resolved - created).total_seconds() / 3600)  # hours
                
                avg_resolution_time = sum(resolution_times) / len(resolution_times)
            
            # Weekly activity
            one_week_ago = datetime.now() - timedelta(days=7)
            issues_closed_last_week = len([
                issue for issue in project_issues
                if issue.resolved_at and datetime.fromisoformat(issue.resolved_at.replace('Z', '+00:00')) >= one_week_ago
            ])
            
            issues_created_last_week = len([
                issue for issue in project_issues
                if datetime.fromisoformat(issue.created_at.replace('Z', '+00:00')) >= one_week_ago
            ])
            
            # Story points
            story_points_completed = sum(
                issue.story_points or 0
                for issue in project_issues
                if issue.status == IssueStatus.CLOSED and issue.story_points
            )
            
            # Team metrics
            assignees = defaultdict(int)
            contributors = set()
            
            for issue in project_issues:
                if issue.assignee:
                    assignees[issue.assignee] += 1
                    contributors.add(issue.assignee)
                if issue.reporter:
                    contributors.add(issue.reporter)
            
            metrics = ProjectMetrics(
                project_name=project_name,
                total_issues=total_issues,
                open_issues=open_issues,
                closed_issues=closed_issues,
                status_breakdown=dict(status_breakdown),
                priority_breakdown=dict(priority_breakdown),
                type_breakdown=dict(type_breakdown),
                avg_resolution_time_hours=avg_resolution_time,
                issues_closed_last_week=issues_closed_last_week,
                issues_created_last_week=issues_created_last_week,
                story_points_completed=story_points_completed,
                active_contributors=list(contributors),
                top_assignees=dict(assignees)
            )
            
            # Store metrics
            self.project_metrics[project_name] = metrics
            await self._store_project_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to generate project metrics: {e}")
            return ProjectMetrics(
                project_name=project_name,
                total_issues=0,
                open_issues=0,
                closed_issues=0
            )
    
    async def _store_project_metrics(self, metrics: ProjectMetrics):
        """Store project metrics to disk"""
        try:
            metrics_file = self.projects_dir / f"{metrics.project_name}_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to store project metrics: {e}")
    
    async def sync_all_configured_providers(self, full_sync: bool = False) -> Dict[str, int]:
        """Sync all configured providers"""
        results = {}
        
        for provider, config in self.providers.items():
            try:
                if provider == TrackerProvider.GITHUB:
                    count = await self.sync_github_issues(config["repository"], full_sync)
                    results[provider.value] = count
                elif provider == TrackerProvider.JIRA:
                    count = await self.sync_jira_issues(config["project_key"], full_sync)
                    results[provider.value] = count
                elif provider == TrackerProvider.GITLAB:
                    # GitLab sync would be implemented similarly
                    results[provider.value] = 0
            except Exception as e:
                logger.error(f"Failed to sync {provider.value}: {e}")
                results[provider.value] = 0
        
        return results


# Global issue tracker sync instance
issue_tracker = IssueTrackerSync()

# Convenience functions
async def setup_github_issues(token: str, repository: str) -> bool:
    """Setup GitHub Issues integration"""
    return await issue_tracker.setup_github_integration(token, repository)

async def setup_jira_issues(url: str, username: str, token: str, project_key: str) -> bool:
    """Setup JIRA integration"""
    return await issue_tracker.setup_jira_integration(url, username, token, project_key)

async def sync_issues(provider: TrackerProvider = None, full_sync: bool = False) -> Dict[str, int]:
    """Sync issues from configured providers"""
    if provider:
        # Sync specific provider
        if provider == TrackerProvider.GITHUB:
            config = issue_tracker.providers.get(provider)
            if config:
                count = await issue_tracker.sync_github_issues(config["repository"], full_sync)
                return {provider.value: count}
        # Add other providers as needed
        return {}
    else:
        # Sync all providers
        return await issue_tracker.sync_all_configured_providers(full_sync)

async def get_project_stats(project_name: str) -> ProjectMetrics:
    """Get project metrics"""
    return await issue_tracker.generate_project_metrics(project_name)

if __name__ == "__main__":
    # Test issue tracker sync
    async def test_issue_tracker():
        print("🎫 Testing Issue Tracker Synchronization")
        
        # Test basic functionality
        print("✅ Issue Tracker Sync system initialized")
        print(f"   Providers supported: {len(TrackerProvider)}")
        print(f"   Issue statuses: {len(IssueStatus)}")
        print(f"   Issue types: {len(IssueType)}")
        
        # Test project metrics with mock data
        metrics = await get_project_stats("test-project")
        print(f"📊 Project metrics generated:")
        print(f"   Total issues: {metrics.total_issues}")
        print(f"   Open issues: {metrics.open_issues}")
        print(f"   Closed issues: {metrics.closed_issues}")
        
        print("✅ Issue Tracker Synchronization ready!")
    
    asyncio.run(test_issue_tracker())