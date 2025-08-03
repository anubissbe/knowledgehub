"""
Central path configuration for KnowledgeHub.
All paths are resolved relative to environment variables or installation directory.
"""
import os
from pathlib import Path

# Get base paths from environment or use defaults
KNOWLEDGEHUB_BASE = Path(os.environ.get('KNOWLEDGEHUB_BASE', Path(__file__).parent.parent))
MEMORY_SYSTEM_BASE = Path(os.environ.get('MEMORY_SYSTEM_BASE', KNOWLEDGEHUB_BASE / 'data' / 'memory-system'))
DATA_BASE = Path(os.environ.get('KNOWLEDGEHUB_DATA', KNOWLEDGEHUB_BASE / 'data'))

# Ensure base directories exist
DATA_BASE.mkdir(parents=True, exist_ok=True)
MEMORY_SYSTEM_BASE.mkdir(parents=True, exist_ok=True)

# Memory system paths
# Point to the actual location of memory-cli
MEMORY_CLI_PATH = os.environ.get('MEMORY_CLI_PATH', '/opt/projects/memory-system/memory-cli')
MEMORY_DATA_PATH = MEMORY_SYSTEM_BASE / 'data'

# Multi-tenant paths
TENANTS_BASE = MEMORY_DATA_PATH / 'tenants'
AUDIT_LOGS_BASE = MEMORY_DATA_PATH / 'audit_logs'

# Incremental loading paths
INCREMENTAL_LOADING_BASE = MEMORY_DATA_PATH / 'incremental_loading'
CONTEXT_CACHE_BASE = MEMORY_DATA_PATH / 'context_cache'

# Sharding paths
SHARDING_BASE = MEMORY_DATA_PATH / 'shards'

# IDE integration paths
IDE_EVENTS_BASE = MEMORY_DATA_PATH / 'ide_events'
IDE_SESSIONS_BASE = MEMORY_DATA_PATH / 'ide_sessions'
IDE_PROJECTS_BASE = MEMORY_DATA_PATH / 'ide_projects'

# Issue tracker paths
ISSUES_BASE = MEMORY_DATA_PATH / 'issues'
PROJECTS_BASE = MEMORY_DATA_PATH / 'projects'
SYNC_LOGS_BASE = MEMORY_DATA_PATH / 'sync_logs'

# Git integration paths
GIT_COMMITS_BASE = MEMORY_DATA_PATH / 'git_commits'
GIT_BRANCHES_BASE = MEMORY_DATA_PATH / 'git_branches'
GIT_REPOS_BASE = MEMORY_DATA_PATH / 'git_repositories'

# CI/CD paths
PIPELINES_BASE = MEMORY_DATA_PATH / 'pipelines'
ARTIFACTS_BASE = MEMORY_DATA_PATH / 'artifacts'
REPORTS_BASE = MEMORY_DATA_PATH / 'reports'

# Ensure all directories exist
for path in [
    MEMORY_DATA_PATH, TENANTS_BASE, AUDIT_LOGS_BASE,
    INCREMENTAL_LOADING_BASE, CONTEXT_CACHE_BASE, SHARDING_BASE,
    IDE_EVENTS_BASE, IDE_SESSIONS_BASE, IDE_PROJECTS_BASE,
    ISSUES_BASE, PROJECTS_BASE, SYNC_LOGS_BASE,
    GIT_COMMITS_BASE, GIT_BRANCHES_BASE, GIT_REPOS_BASE,
    PIPELINES_BASE, ARTIFACTS_BASE, REPORTS_BASE
]:
    path.mkdir(parents=True, exist_ok=True)

# Helper function to get absolute paths
def get_absolute_path(relative_path: str) -> Path:
    """Convert a relative path to absolute path based on KNOWLEDGEHUB_BASE."""
    return KNOWLEDGEHUB_BASE / relative_path

# Export commonly used paths as strings for backward compatibility
ACTUAL_DATA_PATH = str(DATA_BASE)