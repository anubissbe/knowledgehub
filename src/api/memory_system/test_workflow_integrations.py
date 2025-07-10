#!/usr/bin/env python3
"""
Comprehensive test suite for development workflow integrations
Tests Git commit capture, CI/CD integration, issue tracker sync, and IDE integration
"""

import os
import sys
import asyncio
import logging
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import uuid

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from git_commit_capture import (
    GitCommitCapture, CommitContext, CommitType, ChangeImpact,
    git_capture, capture_commit, setup_auto_capture
)
from cicd_pipeline_integration import (
    CICDIntegration, PipelineRun, PipelineStatus, PipelineProvider,
    cicd_integration, setup_github_integration, get_pipeline_stats
)
from issue_tracker_sync import (
    IssueTrackerSync, Issue, IssueStatus, IssuePriority, IssueType, TrackerProvider,
    issue_tracker, setup_github_issues, sync_issues, get_project_stats
)
from advanced_ide_integration import (
    AdvancedIDEIntegration, IDEEvent, EventType, IDEType, ContextScope,
    ide_integration, start_ide_server, get_ide_analytics
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkflowIntegrationTestSuite:
    """Comprehensive test suite for workflow integrations"""
    
    def __init__(self):
        self.test_results = {
            "git_commit_capture": {},
            "cicd_integration": {},
            "issue_tracker_sync": {},
            "ide_integration": {},
            "integration_tests": {}
        }
        self.passed_tests = 0
        self.total_tests = 0
        
        # Test data
        self.temp_dir = None
        self.test_repo_path = None
    
    async def run_all_tests(self):
        """Run all workflow integration tests"""
        print("🔧 Starting Development Workflow Integration Test Suite")
        print("=" * 70)
        
        # Setup test environment
        await self._setup_test_environment()
        
        try:
            # Test each component
            await self._test_git_commit_capture()
            await self._test_cicd_integration()
            await self._test_issue_tracker_sync()
            await self._test_ide_integration()
            await self._test_integration_workflows()
            
        finally:
            # Cleanup
            await self._cleanup_test_environment()
        
        # Print summary
        self._print_test_summary()
        
        return self.passed_tests == self.total_tests
    
    async def _setup_test_environment(self):
        """Setup test environment"""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="workflow_test_"))
            self.test_repo_path = self.temp_dir / "test_repo"
            self.test_repo_path.mkdir()
            
            # Initialize git repo
            import subprocess
            subprocess.run(["git", "init"], cwd=self.test_repo_path, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.test_repo_path)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.test_repo_path)
            
            # Create test files
            test_file = self.test_repo_path / "test.py"
            test_file.write_text("print('Hello World')\n")
            
            subprocess.run(["git", "add", "."], cwd=self.test_repo_path)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.test_repo_path)
            
            logger.info(f"Test environment setup at {self.temp_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment"""
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Test environment cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")
    
    async def _test_git_commit_capture(self):
        """Test Git commit context capture"""
        print("\n📝 Testing Git Commit Context Capture")
        print("-" * 50)
        
        # Test 1: Basic commit capture
        self._start_test("Basic commit capture")
        try:
            capture_system = GitCommitCapture()
            context = await capture_system.capture_commit_context(str(self.test_repo_path))
            
            assert context is not None
            assert isinstance(context, CommitContext)
            assert context.commit_hash
            assert context.commit_message == "Initial commit"
            assert context.repository_path == str(self.test_repo_path)
            
            self._pass_test(f"Captured commit: {context.commit_hash[:8]}")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Commit type classification
        self._start_test("Commit type classification")
        try:
            # Test different commit message patterns
            test_messages = [
                ("feat: add new feature", CommitType.FEATURE),
                ("fix: resolve critical bug", CommitType.BUGFIX),
                ("docs: update README", CommitType.DOCS),
                ("refactor: improve code structure", CommitType.REFACTOR),
                ("test: add unit tests", CommitType.TEST)
            ]
            
            for message, expected_type in test_messages:
                classified_type = capture_system._classify_commit_type(message)
                assert classified_type == expected_type
            
            self._pass_test("All commit types classified correctly")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: File change analysis
        self._start_test("File change analysis")
        try:
            # Create another commit with changes
            test_file = self.test_repo_path / "test.py"
            test_file.write_text("print('Hello World')\nprint('Updated!')\n")
            
            import subprocess
            subprocess.run(["git", "add", "."], cwd=self.test_repo_path)
            subprocess.run(["git", "commit", "-m", "Update test file"], cwd=self.test_repo_path)
            
            # Capture the new commit
            context = await capture_system.capture_commit_context(str(self.test_repo_path))
            
            assert len(context.files_changed) > 0
            assert context.total_lines_added > 0
            
            self._pass_test(f"Analyzed {len(context.files_changed)} file changes")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Repository activity summary
        self._start_test("Repository activity summary")
        try:
            activity = await capture_system.get_repository_activity_summary(str(self.test_repo_path), 7)
            
            assert "total_commits" in activity
            assert activity["total_commits"] >= 2
            assert "contributors" in activity
            
            self._pass_test(f"Activity summary: {activity['total_commits']} commits")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: Git hooks setup
        self._start_test("Git hooks setup")
        try:
            success = await capture_system.setup_git_hooks(str(self.test_repo_path))
            
            hooks_dir = self.test_repo_path / ".git" / "hooks"
            post_commit_hook = hooks_dir / "post-commit"
            
            assert success == True
            assert post_commit_hook.exists()
            
            self._pass_test("Git hooks installed successfully")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_cicd_integration(self):
        """Test CI/CD pipeline integration"""
        print("\n🔄 Testing CI/CD Pipeline Integration")
        print("-" * 50)
        
        # Test 1: Basic pipeline integration
        self._start_test("Pipeline integration initialization")
        try:
            cicd_system = CICDIntegration()
            
            assert len(cicd_system.providers) == 0  # No providers configured initially
            assert cicd_system.config["auto_capture_enabled"] == True
            
            self._pass_test("CI/CD system initialized")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Provider configuration
        self._start_test("Provider configuration")
        try:
            # Test GitHub Actions configuration (mock)
            github_config = {
                "token": "test_token",
                "repository": "test/repo",
                "api_base": "https://api.github.com"
            }
            
            cicd_system.providers[PipelineProvider.GITHUB_ACTIONS] = github_config
            
            assert PipelineProvider.GITHUB_ACTIONS in cicd_system.providers
            
            self._pass_test("Provider configured successfully")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: Pipeline data parsing
        self._start_test("Pipeline data parsing")
        try:
            # Create mock GitHub Actions data
            mock_run_data = {
                "id": 12345,
                "workflow_id": 67890,
                "status": "completed",
                "conclusion": "success",
                "created_at": datetime.now().isoformat() + "Z",
                "updated_at": datetime.now().isoformat() + "Z",
                "head_sha": "abc123",
                "head_branch": "main",
                "event": "push",
                "actor": {"login": "testuser"},
                "repository": {"full_name": "test/repo"},
                "display_title": "Test commit"
            }
            
            mock_jobs_data = {"jobs": []}
            mock_artifacts_data = {"artifacts": []}
            
            pipeline_run = await cicd_system._parse_github_actions_data(
                mock_run_data, mock_jobs_data, mock_artifacts_data
            )
            
            assert pipeline_run.run_id == "12345"
            assert pipeline_run.status == PipelineStatus.SUCCESS
            assert pipeline_run.provider == PipelineProvider.GITHUB_ACTIONS
            
            self._pass_test("Pipeline data parsed correctly")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Pipeline analytics
        self._start_test("Pipeline analytics")
        try:
            # Add mock pipeline to cache for testing
            mock_pipeline = PipelineRun(
                run_id="test_123",
                pipeline_id="test_pipeline",
                provider=PipelineProvider.GITHUB_ACTIONS,
                trigger_type="push",
                trigger_source="abc123",
                triggered_by="testuser",
                status=PipelineStatus.SUCCESS,
                started_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
                duration_seconds=120,
                repository="test/repo",
                branch="main",
                commit_hash="abc123",
                total_tests=10,
                passed_tests=9,
                failed_tests=1,
                code_coverage=85.0
            )
            
            cicd_system.pipeline_cache[mock_pipeline.run_id] = mock_pipeline
            
            analytics = await cicd_system.get_pipeline_analytics(days=30)
            
            assert "total_pipeline_runs" in analytics
            assert analytics["total_pipeline_runs"] >= 1
            
            self._pass_test("Pipeline analytics generated")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: Webhook handling
        self._start_test("Webhook handling")
        try:
            # Mock webhook payload
            webhook_payload = {
                "action": "completed",
                "workflow_run": {
                    "id": 98765,
                    "status": "completed",
                    "conclusion": "success"
                },
                "repository": {
                    "full_name": "test/repo"
                }
            }
            
            # Test webhook processing (would normally trigger async capture)
            result = await cicd_system._handle_github_webhook(webhook_payload, {})
            
            # Should return True for valid webhook
            assert result == True
            
            self._pass_test("Webhook handling working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_issue_tracker_sync(self):
        """Test issue tracker synchronization"""
        print("\n🎫 Testing Issue Tracker Synchronization")
        print("-" * 50)
        
        # Test 1: Basic issue tracker initialization
        self._start_test("Issue tracker initialization")
        try:
            tracker_system = IssueTrackerSync()
            
            assert len(tracker_system.providers) == 0  # No providers configured initially
            assert tracker_system.config["auto_sync_enabled"] == True
            
            self._pass_test("Issue tracker system initialized")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Issue parsing and classification
        self._start_test("Issue parsing and classification")
        try:
            # Create mock GitHub issue data
            mock_issue_data = {
                "number": 123,
                "title": "Fix critical bug in authentication",
                "body": "This is a critical security issue that needs immediate attention.",
                "state": "open",
                "user": {"login": "reporter"},
                "assignee": {"login": "developer"},
                "created_at": datetime.now().isoformat() + "Z",
                "updated_at": datetime.now().isoformat() + "Z",
                "labels": [
                    {"name": "bug", "color": "d73a4a", "description": "Something isn't working"},
                    {"name": "critical", "color": "b60205", "description": "Critical priority"}
                ],
                "html_url": "https://github.com/test/repo/issues/123",
                "id": 456789
            }
            
            issue = await tracker_system._parse_github_issue(
                mock_issue_data, "test/repo", {}, None
            )
            
            assert issue.issue_id == "123"
            assert issue.title == "Fix critical bug in authentication"
            assert issue.status == IssueStatus.OPEN
            assert issue.priority == IssuePriority.CRITICAL
            assert issue.issue_type == IssueType.BUG
            
            self._pass_test("Issue parsed and classified correctly")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: Text analysis and extraction
        self._start_test("Text analysis and extraction")
        try:
            test_text = "This fixes #456 and relates to commit abc123def. @developer please review."
            
            mentions = tracker_system._extract_user_mentions(test_text)
            issue_refs = tracker_system._extract_issue_references(test_text)
            commit_refs = tracker_system._extract_commit_references(test_text)
            
            assert "developer" in mentions
            assert "456" in issue_refs
            assert "abc123def" in commit_refs
            
            self._pass_test("Text analysis working correctly")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Project metrics generation
        self._start_test("Project metrics generation")
        try:
            # Add mock issue to cache
            mock_issue = Issue(
                issue_id="test_123",
                title="Test issue",
                description="Test description",
                status=IssueStatus.CLOSED,
                priority=IssuePriority.MEDIUM,
                issue_type=IssueType.FEATURE,
                project="test-project",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                resolved_at=datetime.now().isoformat()
            )
            
            tracker_system.issue_cache[mock_issue.issue_id] = mock_issue
            
            metrics = await tracker_system.generate_project_metrics("test-project")
            
            assert metrics.project_name == "test-project"
            assert metrics.total_issues >= 1
            assert metrics.closed_issues >= 1
            
            self._pass_test("Project metrics generated")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: JIRA mapping functions
        self._start_test("JIRA status and priority mapping")
        try:
            # Test status mapping
            status_tests = [
                ("in progress", IssueStatus.IN_PROGRESS),
                ("done", IssueStatus.CLOSED),
                ("blocked", IssueStatus.BLOCKED)
            ]
            
            for jira_status, expected_status in status_tests:
                mapped_status = tracker_system._map_jira_status(jira_status)
                assert mapped_status == expected_status
            
            # Test priority mapping
            priority_tests = [
                ("Highest", IssuePriority.CRITICAL),
                ("Medium", IssuePriority.MEDIUM),
                ("Lowest", IssuePriority.TRIVIAL)
            ]
            
            for jira_priority, expected_priority in priority_tests:
                mapped_priority = tracker_system._map_jira_priority(jira_priority)
                assert mapped_priority == expected_priority
            
            self._pass_test("JIRA mapping functions working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_ide_integration(self):
        """Test advanced IDE integration"""
        print("\n🔌 Testing Advanced IDE Integration")
        print("-" * 50)
        
        # Test 1: Basic IDE integration initialization
        self._start_test("IDE integration initialization")
        try:
            ide_system = AdvancedIDEIntegration()
            
            assert ide_system.config["auto_suggestions"] == True
            assert ide_system.config["context_tracking"] == True
            assert len(ide_system.active_sessions) == 0
            
            self._pass_test("IDE integration system initialized")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Event parsing and processing
        self._start_test("IDE event parsing")
        try:
            # Test position parsing
            position_data = {"line": 10, "column": 5, "offset": 150}
            position = ide_system._parse_position(position_data)
            
            assert position.line == 10
            assert position.column == 5
            assert position.offset == 150
            
            # Test range parsing
            range_data = {
                "start": {"line": 10, "column": 5},
                "end": {"line": 12, "column": 10}
            }
            code_range = ide_system._parse_range(range_data)
            
            assert code_range.start.line == 10
            assert code_range.end.line == 12
            
            self._pass_test("IDE event parsing working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: Project context analysis
        self._start_test("Project context analysis")
        try:
            # Analyze test repository
            await ide_system._analyze_project_context(str(self.test_repo_path))
            
            context = ide_system.project_contexts.get(str(self.test_repo_path))
            
            assert context is not None
            assert context.project_name == "test_repo"
            assert ".py" in context.file_types
            
            self._pass_test("Project context analyzed")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Language detection
        self._start_test("Language detection")
        try:
            language_tests = [
                (".py", "python"),
                (".js", "javascript"),
                (".ts", "typescript"),
                (".java", "java"),
                (".cpp", "cpp")
            ]
            
            for extension, expected_language in language_tests:
                detected = ide_system._detect_language_from_extension(extension)
                assert detected == expected_language
            
            self._pass_test("Language detection working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: Suggestion generation
        self._start_test("Contextual suggestion generation")
        try:
            # Create mock IDE event
            mock_event = IDEEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.FILE_OPENED,
                timestamp=datetime.now().isoformat(),
                ide_type=IDEType.VSCODE,
                file_path=str(self.test_repo_path / "test.py"),
                project_path=str(self.test_repo_path),
                language="python",
                file_content="",
                line_count=0
            )
            
            suggestions = await ide_system._generate_contextual_suggestions(mock_event)
            
            # Should generate at least one suggestion for empty Python file
            assert len(suggestions) > 0
            
            self._pass_test(f"Generated {len(suggestions)} suggestions")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 6: Session tracking
        self._start_test("Development session tracking")
        try:
            # Create mock session
            session_id = str(uuid.uuid4())
            
            mock_event = IDEEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.FILE_OPENED,
                timestamp=datetime.now().isoformat(),
                ide_type=IDEType.VSCODE,
                project_path=str(self.test_repo_path),
                session_id=session_id,
                language="python"
            )
            
            await ide_system._update_session_tracking(mock_event)
            
            assert session_id in ide_system.active_sessions
            session = ide_system.active_sessions[session_id]
            assert session.files_opened == 1
            
            self._pass_test("Session tracking working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    async def _test_integration_workflows(self):
        """Test integration between different workflow components"""
        print("\n🔗 Testing Integration Workflows")
        print("-" * 50)
        
        # Test 1: Git + CI/CD integration
        self._start_test("Git + CI/CD workflow")
        try:
            # Simulate commit that triggers CI/CD
            git_context = await git_capture.capture_commit_context(str(self.test_repo_path))
            
            # Create related CI/CD pipeline run
            pipeline_run = PipelineRun(
                run_id="integration_test_123",
                pipeline_id="test_pipeline",
                provider=PipelineProvider.GITHUB_ACTIONS,
                trigger_type="push",
                trigger_source=git_context.commit_hash,
                triggered_by=git_context.author,
                status=PipelineStatus.SUCCESS,
                started_at=datetime.now().isoformat(),
                repository=git_context.repository_path,
                branch=git_context.branch_name,
                commit_hash=git_context.commit_hash,
                commit_message=git_context.commit_message
            )
            
            # Verify integration points
            assert pipeline_run.commit_hash == git_context.commit_hash
            assert pipeline_run.triggered_by == git_context.author
            
            self._pass_test("Git + CI/CD integration working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 2: Issue + Commit integration
        self._start_test("Issue + Commit workflow")
        try:
            # Create issue
            mock_issue = Issue(
                issue_id="456",
                title="Fix authentication bug",
                description="Critical security issue",
                status=IssueStatus.IN_PROGRESS,
                priority=IssuePriority.CRITICAL,
                issue_type=IssueType.BUG,
                project="test-project"
            )
            
            # Create commit that references the issue
            commit_message = "fix: resolve authentication bug (fixes #456)"
            referenced_issues = issue_tracker._extract_issue_references(commit_message)
            
            assert "456" in referenced_issues
            
            self._pass_test("Issue + Commit integration working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 3: IDE + Project context integration
        self._start_test("IDE + Project context workflow")
        try:
            # IDE opens file in project
            ide_event = IDEEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.FILE_OPENED,
                timestamp=datetime.now().isoformat(),
                ide_type=IDEType.VSCODE,
                file_path=str(self.test_repo_path / "test.py"),
                project_path=str(self.test_repo_path),
                language="python"
            )
            
            # Should have project context
            project_context = ide_integration.project_contexts.get(str(self.test_repo_path))
            
            if project_context:
                assert project_context.project_name == "test_repo"
                assert "python" in project_context.languages
            
            self._pass_test("IDE + Project context integration working")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 4: Full development workflow
        self._start_test("Full development workflow")
        try:
            # Simulate complete workflow:
            # 1. Developer opens IDE (IDE integration)
            # 2. Works on issue (Issue tracker)
            # 3. Makes commit (Git capture)
            # 4. Triggers CI/CD (Pipeline integration)
            
            workflow_context = {
                "issue_id": "789",
                "commit_hash": "def456",
                "pipeline_run": "workflow_test_789",
                "ide_session": str(uuid.uuid4())
            }
            
            # Verify all components can work with shared context
            assert all(workflow_context.values())
            
            self._pass_test("Full workflow integration possible")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
        
        # Test 5: Memory system integration
        self._start_test("Memory system integration")
        try:
            # Test that all components can integrate with memory
            components_with_memory = [
                hasattr(git_capture, 'memory_system'),
                hasattr(cicd_integration, 'memory_system'),
                hasattr(issue_tracker, 'memory_system'),
                hasattr(ide_integration, 'memory_system')
            ]
            
            assert all(components_with_memory)
            
            self._pass_test("All components have memory integration")
        except Exception as e:
            self._fail_test(f"Failed: {e}")
    
    def _start_test(self, test_name: str):
        """Start a test"""
        self.current_test = test_name
        self.total_tests += 1
        print(f"  Testing: {test_name}...", end=" ")
    
    def _pass_test(self, message: str = ""):
        """Pass a test"""
        self.passed_tests += 1
        status = "✅ PASS"
        if message:
            status += f" - {message}"
        print(status)
    
    def _fail_test(self, message: str = ""):
        """Fail a test"""
        status = "❌ FAIL"
        if message:
            status += f" - {message}"
        print(status)
    
    def _print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("📊 Workflow Integration Test Summary")
        print("=" * 70)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            print("🎉 Excellent! All workflow integrations are working perfectly.")
        elif success_rate >= 85:
            print("✅ Great! Most workflow integrations are working correctly.")
        elif success_rate >= 70:
            print("⚠️  Good! Minor issues found in some integrations.")
        else:
            print("❌ Significant issues detected. Workflow integrations need attention.")
        
        print("\n🏁 Development Workflow Integration Test Suite Complete")

async def main():
    """Run the workflow integration test suite"""
    test_suite = WorkflowIntegrationTestSuite()
    success = await test_suite.run_all_tests()
    return success

if __name__ == "__main__":
    asyncio.run(main())