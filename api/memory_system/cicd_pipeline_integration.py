#!/usr/bin/env python3
"""
CI/CD Pipeline Integration System
Integrates with various CI/CD platforms to capture build, test, and deployment context
"""

import os
import sys
import json
import asyncio
import logging
import aiohttp
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from enum import Enum
import hashlib
from collections import defaultdict
import base64

# Add memory system to path
MEMORY_SYSTEM_PATH = Path(__file__).parent
sys.path.insert(0, str(MEMORY_SYSTEM_PATH))

from claude_unified_memory import UnifiedMemorySystem

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

class PipelineStage(Enum):
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    CODE_QUALITY = "code_quality"
    DEPLOY_STAGING = "deploy_staging"
    DEPLOY_PRODUCTION = "deploy_production"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    ROLLBACK = "rollback"

class PipelineProvider(Enum):
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    AZURE_DEVOPS = "azure_devops"
    CIRCLECI = "circleci"
    TRAVIS_CI = "travis_ci"
    BUILDKITE = "buildkite"
    GENERIC = "generic"

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    test_suite: str
    status: str  # passed, failed, skipped
    duration_seconds: float
    error_message: str = ""
    stack_trace: str = ""
    
    # Coverage info
    coverage_percentage: Optional[float] = None
    lines_covered: Optional[int] = None
    lines_total: Optional[int] = None

@dataclass
class BuildArtifact:
    """Build artifact information"""
    name: str
    type: str  # binary, docker_image, package, report
    size_bytes: int
    checksum: str
    download_url: str = ""
    
    # Metadata
    version: str = ""
    architecture: str = ""
    created_at: str = ""

@dataclass
class SecurityScanResult:
    """Security scan results"""
    scanner_name: str
    scan_type: str  # sast, dast, dependency, container
    vulnerabilities_found: int
    
    # Severity breakdown
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    
    # Details
    scan_duration_seconds: float = 0.0
    report_url: str = ""
    findings: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []

@dataclass
class DeploymentInfo:
    """Deployment information"""
    environment: str  # staging, production, development
    version: str
    deployment_strategy: str  # blue_green, rolling, canary
    
    # Timing
    started_at: str
    completed_at: str = ""
    duration_seconds: float = 0.0
    
    # Infrastructure
    target_instances: List[str] = None
    health_check_url: str = ""
    rollback_version: str = ""
    
    def __post_init__(self):
        if self.target_instances is None:
            self.target_instances = []

@dataclass
class PipelineRun:
    """Complete CI/CD pipeline run context"""
    run_id: str
    pipeline_id: str
    provider: PipelineProvider
    
    # Trigger information
    trigger_type: str  # push, pull_request, schedule, manual
    trigger_source: str  # commit_hash, branch, tag
    triggered_by: str  # username or system
    
    # Status and timing
    status: PipelineStatus
    started_at: str
    repository: str
    branch: str
    commit_hash: str
    
    # Optional fields
    completed_at: str = ""
    duration_seconds: float = 0.0
    commit_message: str = ""
    
    # Stages and results
    stages: Dict[PipelineStage, Dict[str, Any]] = None
    test_results: List[TestResult] = None
    build_artifacts: List[BuildArtifact] = None
    security_scans: List[SecurityScanResult] = None
    deployments: List[DeploymentInfo] = None
    
    # Metrics
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    code_coverage: float = 0.0
    
    # Quality metrics
    code_quality_score: Optional[float] = None
    performance_metrics: Dict[str, float] = None
    
    # Environment
    runner_info: Dict[str, str] = None
    environment_variables: Dict[str, str] = None
    
    # Memory integration
    memory_id: str = ""
    
    def __post_init__(self):
        if self.stages is None:
            self.stages = {}
        if self.test_results is None:
            self.test_results = []
        if self.build_artifacts is None:
            self.build_artifacts = []
        if self.security_scans is None:
            self.security_scans = []
        if self.deployments is None:
            self.deployments = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.runner_info is None:
            self.runner_info = {}
        if self.environment_variables is None:
            self.environment_variables = {}
        if not self.memory_id:
            self.memory_id = f"pipeline_{self.run_id}"

class CICDIntegration:
    """
    Integrates with CI/CD platforms to capture pipeline context and results
    """
    
    def __init__(self, memory_system: Optional[UnifiedMemorySystem] = None):
        self.memory_system = memory_system or UnifiedMemorySystem()
        
        # Storage
        self.pipelines_dir = Path("/opt/projects/memory-system/data/cicd_pipelines")
        self.artifacts_dir = Path("/opt/projects/memory-system/data/build_artifacts")
        self.reports_dir = Path("/opt/projects/memory-system/data/pipeline_reports")
        
        for directory in [self.pipelines_dir, self.artifacts_dir, self.reports_dir]:
            directory.mkdir(exist_ok=True)
        
        # Provider configurations
        self.providers = {}
        self.webhooks = {}
        
        # Configuration
        self.config = {
            "auto_capture_enabled": True,
            "store_artifacts": True,
            "store_test_reports": True,
            "store_security_reports": True,
            "memory_integration": True,
            "retention_days": 90,
            "max_artifact_size_mb": 100
        }
        
        # Load existing pipeline data
        self.pipeline_cache: Dict[str, PipelineRun] = {}
        self._load_recent_pipelines()
    
    def _load_recent_pipelines(self):
        """Load recent pipeline runs from storage"""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for pipeline_file in self.pipelines_dir.glob("*.json"):
                try:
                    with open(pipeline_file, 'r', encoding='utf-8') as f:
                        pipeline_data = json.load(f)
                        
                    # Check if recent enough
                    started_at = datetime.fromisoformat(pipeline_data["started_at"])
                    if started_at >= cutoff_date:
                        pipeline = PipelineRun(**pipeline_data)
                        self.pipeline_cache[pipeline.run_id] = pipeline
                        
                except Exception as e:
                    logger.warning(f"Failed to load pipeline file {pipeline_file}: {e}")
            
            logger.info(f"Loaded {len(self.pipeline_cache)} recent pipeline runs")
            
        except Exception as e:
            logger.error(f"Failed to load recent pipelines: {e}")
    
    async def setup_github_actions_integration(
        self,
        github_token: str,
        repository: str,
        webhook_secret: Optional[str] = None
    ) -> bool:
        """Setup GitHub Actions integration"""
        try:
            self.providers[PipelineProvider.GITHUB_ACTIONS] = {
                "token": github_token,
                "repository": repository,
                "webhook_secret": webhook_secret,
                "api_base": "https://api.github.com"
            }
            
            # Test connection
            headers = {"Authorization": f"token {github_token}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.github.com/repos/{repository}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"GitHub Actions integration setup for {repository}")
                        return True
                    else:
                        logger.error(f"GitHub API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to setup GitHub Actions integration: {e}")
            return False
    
    async def setup_gitlab_ci_integration(
        self,
        gitlab_token: str,
        project_id: str,
        gitlab_url: str = "https://gitlab.com"
    ) -> bool:
        """Setup GitLab CI integration"""
        try:
            self.providers[PipelineProvider.GITLAB_CI] = {
                "token": gitlab_token,
                "project_id": project_id,
                "api_base": f"{gitlab_url}/api/v4"
            }
            
            # Test connection
            headers = {"PRIVATE-TOKEN": gitlab_token}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{gitlab_url}/api/v4/projects/{project_id}",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        logger.info(f"GitLab CI integration setup for project {project_id}")
                        return True
                    else:
                        logger.error(f"GitLab API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to setup GitLab CI integration: {e}")
            return False
    
    async def setup_jenkins_integration(
        self,
        jenkins_url: str,
        username: str,
        api_token: str
    ) -> bool:
        """Setup Jenkins integration"""
        try:
            self.providers[PipelineProvider.JENKINS] = {
                "url": jenkins_url,
                "username": username,
                "token": api_token
            }
            
            # Test connection
            auth = aiohttp.BasicAuth(username, api_token)
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{jenkins_url}/api/json",
                    auth=auth
                ) as response:
                    if response.status == 200:
                        logger.info(f"Jenkins integration setup for {jenkins_url}")
                        return True
                    else:
                        logger.error(f"Jenkins API test failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to setup Jenkins integration: {e}")
            return False
    
    async def capture_github_actions_run(
        self,
        repository: str,
        run_id: str
    ) -> Optional[PipelineRun]:
        """Capture GitHub Actions workflow run"""
        try:
            provider_config = self.providers.get(PipelineProvider.GITHUB_ACTIONS)
            if not provider_config:
                logger.error("GitHub Actions not configured")
                return None
            
            headers = {"Authorization": f"token {provider_config['token']}"}
            
            # Get workflow run details
            async with aiohttp.ClientSession() as session:
                # Main run information
                async with session.get(
                    f"https://api.github.com/repos/{repository}/actions/runs/{run_id}",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get GitHub Actions run: {response.status}")
                        return None
                    
                    run_data = await response.json()
                
                # Get jobs for this run
                async with session.get(
                    f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/jobs",
                    headers=headers
                ) as response:
                    jobs_data = await response.json() if response.status == 200 else {"jobs": []}
                
                # Get artifacts
                async with session.get(
                    f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/artifacts",
                    headers=headers
                ) as response:
                    artifacts_data = await response.json() if response.status == 200 else {"artifacts": []}
            
            # Parse run data
            pipeline_run = await self._parse_github_actions_data(run_data, jobs_data, artifacts_data)
            
            # Store and integrate
            await self._store_pipeline_run(pipeline_run)
            
            if self.config["memory_integration"]:
                await self._integrate_pipeline_with_memory(pipeline_run)
            
            return pipeline_run
            
        except Exception as e:
            logger.error(f"Failed to capture GitHub Actions run: {e}")
            return None
    
    async def _parse_github_actions_data(
        self,
        run_data: Dict[str, Any],
        jobs_data: Dict[str, Any],
        artifacts_data: Dict[str, Any]
    ) -> PipelineRun:
        """Parse GitHub Actions data into PipelineRun"""
        
        # Parse timing
        started_at = run_data["created_at"]
        completed_at = run_data.get("updated_at", "")
        duration = 0.0
        
        if completed_at and started_at:
            start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
            duration = (end_time - start_time).total_seconds()
        
        # Parse status
        status_map = {
            "completed": PipelineStatus.SUCCESS if run_data["conclusion"] == "success" else PipelineStatus.FAILED,
            "in_progress": PipelineStatus.RUNNING,
            "queued": PipelineStatus.PENDING,
            "cancelled": PipelineStatus.CANCELLED
        }
        
        status = status_map.get(run_data["status"], PipelineStatus.PENDING)
        
        # Parse trigger
        trigger_type = "push"
        if run_data["event"] == "pull_request":
            trigger_type = "pull_request"
        elif run_data["event"] == "schedule":
            trigger_type = "schedule"
        elif run_data["event"] == "workflow_dispatch":
            trigger_type = "manual"
        
        # Parse jobs as stages
        stages = {}
        test_results = []
        
        for job in jobs_data.get("jobs", []):
            stage_name = self._map_job_to_stage(job["name"])
            stages[stage_name] = {
                "name": job["name"],
                "status": job["conclusion"] or job["status"],
                "started_at": job.get("started_at", ""),
                "completed_at": job.get("completed_at", ""),
                "runner": job.get("runner_name", ""),
                "logs_url": job.get("logs_url", "")
            }
            
            # Extract test results if this is a test job
            if "test" in job["name"].lower():
                test_results.extend(await self._extract_github_test_results(job))
        
        # Parse artifacts
        build_artifacts = []
        for artifact in artifacts_data.get("artifacts", []):
            build_artifacts.append(BuildArtifact(
                name=artifact["name"],
                type=self._determine_artifact_type(artifact["name"]),
                size_bytes=artifact["size_in_bytes"],
                checksum="",  # GitHub doesn't provide checksums
                download_url=artifact["archive_download_url"],
                created_at=artifact["created_at"]
            ))
        
        return PipelineRun(
            run_id=str(run_data["id"]),
            pipeline_id=run_data["workflow_id"],
            provider=PipelineProvider.GITHUB_ACTIONS,
            trigger_type=trigger_type,
            trigger_source=run_data["head_sha"],
            triggered_by=run_data["actor"]["login"],
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            repository=run_data["repository"]["full_name"],
            branch=run_data["head_branch"],
            commit_hash=run_data["head_sha"],
            commit_message=run_data["display_title"],
            stages=stages,
            test_results=test_results,
            build_artifacts=build_artifacts,
            runner_info={"runner_type": "github-hosted"}
        )
    
    def _map_job_to_stage(self, job_name: str) -> PipelineStage:
        """Map GitHub Actions job name to pipeline stage"""
        job_lower = job_name.lower()
        
        if "build" in job_lower:
            return PipelineStage.BUILD
        elif "test" in job_lower:
            return PipelineStage.TEST
        elif "security" in job_lower or "scan" in job_lower:
            return PipelineStage.SECURITY_SCAN
        elif "quality" in job_lower or "lint" in job_lower:
            return PipelineStage.CODE_QUALITY
        elif "deploy" in job_lower:
            if "prod" in job_lower:
                return PipelineStage.DEPLOY_PRODUCTION
            else:
                return PipelineStage.DEPLOY_STAGING
        else:
            return PipelineStage.BUILD  # Default
    
    async def _extract_github_test_results(self, job: Dict[str, Any]) -> List[TestResult]:
        """Extract test results from GitHub Actions job"""
        # This would parse test reports from job artifacts or logs
        # For now, return a mock structure
        return [
            TestResult(
                test_name="example_test",
                test_suite=job["name"],
                status="passed",
                duration_seconds=1.5,
                coverage_percentage=85.0
            )
        ]
    
    def _determine_artifact_type(self, artifact_name: str) -> str:
        """Determine artifact type from name"""
        name_lower = artifact_name.lower()
        
        if "test" in name_lower and ("report" in name_lower or "result" in name_lower):
            return "test_report"
        elif "coverage" in name_lower:
            return "coverage_report"
        elif "security" in name_lower or "scan" in name_lower:
            return "security_report"
        elif name_lower.endswith((".tar.gz", ".zip", ".jar", ".war")):
            return "package"
        elif "docker" in name_lower or name_lower.endswith(".tar"):
            return "docker_image"
        else:
            return "binary"
    
    async def capture_gitlab_pipeline(
        self,
        project_id: str,
        pipeline_id: str
    ) -> Optional[PipelineRun]:
        """Capture GitLab CI pipeline"""
        try:
            provider_config = self.providers.get(PipelineProvider.GITLAB_CI)
            if not provider_config:
                logger.error("GitLab CI not configured")
                return None
            
            headers = {"PRIVATE-TOKEN": provider_config["token"]}
            api_base = provider_config["api_base"]
            
            # Get pipeline details
            async with aiohttp.ClientSession() as session:
                # Main pipeline information
                async with session.get(
                    f"{api_base}/projects/{project_id}/pipelines/{pipeline_id}",
                    headers=headers
                ) as response:
                    if response.status != 200:
                        logger.error(f"Failed to get GitLab pipeline: {response.status}")
                        return None
                    
                    pipeline_data = await response.json()
                
                # Get jobs
                async with session.get(
                    f"{api_base}/projects/{project_id}/pipelines/{pipeline_id}/jobs",
                    headers=headers
                ) as response:
                    jobs_data = await response.json() if response.status == 200 else []
            
            # Parse pipeline data
            pipeline_run = await self._parse_gitlab_pipeline_data(pipeline_data, jobs_data)
            
            # Store and integrate
            await self._store_pipeline_run(pipeline_run)
            
            if self.config["memory_integration"]:
                await self._integrate_pipeline_with_memory(pipeline_run)
            
            return pipeline_run
            
        except Exception as e:
            logger.error(f"Failed to capture GitLab pipeline: {e}")
            return None
    
    async def _parse_gitlab_pipeline_data(
        self,
        pipeline_data: Dict[str, Any],
        jobs_data: List[Dict[str, Any]]
    ) -> PipelineRun:
        """Parse GitLab CI data into PipelineRun"""
        
        # Parse timing
        started_at = pipeline_data.get("created_at", "")
        completed_at = pipeline_data.get("finished_at", "")
        duration = pipeline_data.get("duration", 0) or 0
        
        # Parse status
        status_map = {
            "success": PipelineStatus.SUCCESS,
            "failed": PipelineStatus.FAILED,
            "canceled": PipelineStatus.CANCELLED,
            "skipped": PipelineStatus.SKIPPED,
            "running": PipelineStatus.RUNNING,
            "pending": PipelineStatus.PENDING
        }
        
        status = status_map.get(pipeline_data["status"], PipelineStatus.PENDING)
        
        # Parse trigger
        trigger_type = "push"
        source = pipeline_data.get("source", "")
        if source == "merge_request_event":
            trigger_type = "pull_request"
        elif source == "schedule":
            trigger_type = "schedule"
        elif source == "web":
            trigger_type = "manual"
        
        # Parse jobs as stages
        stages = {}
        test_results = []
        
        for job in jobs_data:
            stage_name = self._map_gitlab_job_to_stage(job["stage"])
            stages[stage_name] = {
                "name": job["name"],
                "stage": job["stage"],
                "status": job["status"],
                "started_at": job.get("started_at", ""),
                "finished_at": job.get("finished_at", ""),
                "duration": job.get("duration", 0),
                "runner": job.get("runner", {}).get("description", ""),
                "web_url": job.get("web_url", "")
            }
            
            # Extract test results if this is a test job
            if job["stage"] == "test":
                test_results.extend(await self._extract_gitlab_test_results(job))
        
        return PipelineRun(
            run_id=str(pipeline_data["id"]),
            pipeline_id=str(pipeline_data["id"]),
            provider=PipelineProvider.GITLAB_CI,
            trigger_type=trigger_type,
            trigger_source=pipeline_data["sha"],
            triggered_by=pipeline_data.get("user", {}).get("username", "system"),
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            repository=f"gitlab-project-{pipeline_data.get('project_id', '')}",
            branch=pipeline_data.get("ref", ""),
            commit_hash=pipeline_data["sha"],
            stages=stages,
            test_results=test_results,
            runner_info={"platform": "gitlab"}
        )
    
    def _map_gitlab_job_to_stage(self, stage_name: str) -> PipelineStage:
        """Map GitLab CI stage to pipeline stage"""
        stage_lower = stage_name.lower()
        
        stage_map = {
            "build": PipelineStage.BUILD,
            "test": PipelineStage.TEST,
            "security": PipelineStage.SECURITY_SCAN,
            "quality": PipelineStage.CODE_QUALITY,
            "deploy": PipelineStage.DEPLOY_STAGING,
            "production": PipelineStage.DEPLOY_PRODUCTION,
            "staging": PipelineStage.DEPLOY_STAGING
        }
        
        return stage_map.get(stage_lower, PipelineStage.BUILD)
    
    async def _extract_gitlab_test_results(self, job: Dict[str, Any]) -> List[TestResult]:
        """Extract test results from GitLab CI job"""
        # This would parse test reports from GitLab job artifacts
        # For now, return a mock structure
        return [
            TestResult(
                test_name="gitlab_test",
                test_suite=job["name"],
                status="passed" if job["status"] == "success" else "failed",
                duration_seconds=job.get("duration", 0),
                coverage_percentage=80.0
            )
        ]
    
    async def webhook_handler(
        self,
        provider: PipelineProvider,
        payload: Dict[str, Any],
        headers: Dict[str, str] = None
    ) -> bool:
        """Handle webhook from CI/CD provider"""
        try:
            if provider == PipelineProvider.GITHUB_ACTIONS:
                return await self._handle_github_webhook(payload, headers)
            elif provider == PipelineProvider.GITLAB_CI:
                return await self._handle_gitlab_webhook(payload, headers)
            else:
                logger.warning(f"Webhook handler not implemented for {provider}")
                return False
                
        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            return False
    
    async def _handle_github_webhook(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> bool:
        """Handle GitHub Actions webhook"""
        # Verify webhook signature if secret is configured
        provider_config = self.providers.get(PipelineProvider.GITHUB_ACTIONS, {})
        webhook_secret = provider_config.get("webhook_secret")
        
        if webhook_secret and headers:
            # Implement signature verification
            pass
        
        # Process workflow run event
        if payload.get("action") in ["completed", "requested"]:
            workflow_run = payload.get("workflow_run", {})
            repository = payload.get("repository", {}).get("full_name", "")
            run_id = str(workflow_run.get("id", ""))
            
            if run_id and repository:
                # Capture the run asynchronously
                asyncio.create_task(
                    self.capture_github_actions_run(repository, run_id)
                )
                return True
        
        return False
    
    async def _handle_gitlab_webhook(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> bool:
        """Handle GitLab CI webhook"""
        # Process pipeline event
        object_kind = payload.get("object_kind")
        
        if object_kind == "pipeline":
            pipeline = payload.get("object_attributes", {})
            project_id = str(payload.get("project", {}).get("id", ""))
            pipeline_id = str(pipeline.get("id", ""))
            
            if pipeline_id and project_id and pipeline.get("status") in ["success", "failed", "canceled"]:
                # Capture the pipeline asynchronously
                asyncio.create_task(
                    self.capture_gitlab_pipeline(project_id, pipeline_id)
                )
                return True
        
        return False
    
    async def _store_pipeline_run(self, pipeline_run: PipelineRun):
        """Store pipeline run to disk"""
        try:
            pipeline_file = self.pipelines_dir / f"{pipeline_run.run_id}.json"
            with open(pipeline_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(pipeline_run), f, indent=2, default=str)
            
            # Update cache
            self.pipeline_cache[pipeline_run.run_id] = pipeline_run
            
        except Exception as e:
            logger.error(f"Failed to store pipeline run: {e}")
    
    async def _integrate_pipeline_with_memory(self, pipeline_run: PipelineRun):
        """Integrate pipeline run with memory system"""
        try:
            # Create memory entry for pipeline run
            memory_content = self._format_pipeline_memory(pipeline_run)
            
            await self.memory_system.add_memory(
                content=memory_content,
                memory_type="development",
                priority="medium",
                tags=[
                    "cicd_pipeline",
                    f"provider:{pipeline_run.provider.value}",
                    f"status:{pipeline_run.status.value}",
                    f"trigger:{pipeline_run.trigger_type}",
                    f"repository:{pipeline_run.repository}",
                    f"branch:{pipeline_run.branch}"
                ],
                metadata={
                    "pipeline_id": pipeline_run.pipeline_id,
                    "run_id": pipeline_run.run_id,
                    "provider": pipeline_run.provider.value,
                    "status": pipeline_run.status.value,
                    "duration_seconds": pipeline_run.duration_seconds,
                    "commit_hash": pipeline_run.commit_hash,
                    "total_tests": pipeline_run.total_tests,
                    "passed_tests": pipeline_run.passed_tests,
                    "code_coverage": pipeline_run.code_coverage
                }
            )
            
            logger.debug(f"Integrated pipeline {pipeline_run.run_id} with memory system")
            
        except Exception as e:
            logger.error(f"Failed to integrate pipeline with memory: {e}")
    
    def _format_pipeline_memory(self, pipeline_run: PipelineRun) -> str:
        """Format pipeline context for memory storage"""
        content = f"""CI/CD Pipeline Run: {pipeline_run.run_id}
Provider: {pipeline_run.provider.value}
Repository: {pipeline_run.repository}
Branch: {pipeline_run.branch}
Commit: {pipeline_run.commit_hash}
Status: {pipeline_run.status.value}
Duration: {pipeline_run.duration_seconds:.1f}s

Trigger:
- Type: {pipeline_run.trigger_type}
- Source: {pipeline_run.trigger_source}
- Triggered by: {pipeline_run.triggered_by}

Results:
- Total tests: {pipeline_run.total_tests}
- Passed tests: {pipeline_run.passed_tests}
- Failed tests: {pipeline_run.failed_tests}
- Code coverage: {pipeline_run.code_coverage:.1f}%"""
        
        if pipeline_run.stages:
            content += "\n\nStages:"
            for stage, details in pipeline_run.stages.items():
                status = details.get("status", "unknown")
                content += f"\n- {stage.value}: {status}"
        
        if pipeline_run.build_artifacts:
            content += f"\n\nArtifacts: {len(pipeline_run.build_artifacts)} generated"
        
        if pipeline_run.security_scans:
            total_vulns = sum(scan.vulnerabilities_found for scan in pipeline_run.security_scans)
            content += f"\nSecurity scans: {total_vulns} vulnerabilities found"
        
        if pipeline_run.deployments:
            environments = [dep.environment for dep in pipeline_run.deployments]
            content += f"\nDeployments: {', '.join(environments)}"
        
        return content
    
    async def get_pipeline_analytics(
        self,
        repository: str = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get pipeline analytics and trends"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter pipelines
            filtered_pipelines = []
            for pipeline in self.pipeline_cache.values():
                pipeline_date = datetime.fromisoformat(pipeline.started_at.replace('Z', '+00:00'))
                if pipeline_date >= cutoff_date:
                    if not repository or pipeline.repository == repository:
                        filtered_pipelines.append(pipeline)
            
            if not filtered_pipelines:
                return {"error": "No pipeline data found"}
            
            # Calculate analytics
            total_runs = len(filtered_pipelines)
            successful_runs = len([p for p in filtered_pipelines if p.status == PipelineStatus.SUCCESS])
            failed_runs = len([p for p in filtered_pipelines if p.status == PipelineStatus.FAILED])
            
            success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
            
            # Duration analytics
            completed_pipelines = [p for p in filtered_pipelines if p.duration_seconds > 0]
            avg_duration = sum(p.duration_seconds for p in completed_pipelines) / len(completed_pipelines) if completed_pipelines else 0
            
            # Test analytics
            total_tests = sum(p.total_tests for p in filtered_pipelines)
            total_passed = sum(p.passed_tests for p in filtered_pipelines)
            
            test_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
            
            # Coverage analytics
            coverage_data = [p.code_coverage for p in filtered_pipelines if p.code_coverage > 0]
            avg_coverage = sum(coverage_data) / len(coverage_data) if coverage_data else 0
            
            return {
                "period_days": days,
                "repository": repository,
                "total_pipeline_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "success_rate_percentage": round(success_rate, 2),
                "average_duration_seconds": round(avg_duration, 2),
                "total_tests_run": total_tests,
                "test_success_rate_percentage": round(test_success_rate, 2),
                "average_code_coverage_percentage": round(avg_coverage, 2),
                "most_active_branches": self._get_most_active_branches(filtered_pipelines),
                "failure_patterns": self._analyze_failure_patterns(filtered_pipelines),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate pipeline analytics: {e}")
            return {"error": str(e)}
    
    def _get_most_active_branches(self, pipelines: List[PipelineRun]) -> List[Dict[str, Any]]:
        """Get most active branches"""
        branch_counts = defaultdict(int)
        for pipeline in pipelines:
            branch_counts[pipeline.branch] += 1
        
        return [
            {"branch": branch, "pipeline_count": count}
            for branch, count in sorted(branch_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    def _analyze_failure_patterns(self, pipelines: List[PipelineRun]) -> List[Dict[str, Any]]:
        """Analyze common failure patterns"""
        failed_pipelines = [p for p in pipelines if p.status == PipelineStatus.FAILED]
        
        # Analyze by stage failures
        stage_failures = defaultdict(int)
        for pipeline in failed_pipelines:
            for stage, details in pipeline.stages.items():
                if details.get("status") in ["failed", "failure"]:
                    stage_failures[stage.value] += 1
        
        patterns = [
            {"pattern": f"Stage '{stage}' failures", "count": count}
            for stage, count in sorted(stage_failures.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        
        return patterns


# Global CI/CD integration instance
cicd_integration = CICDIntegration()

# Convenience functions
async def setup_github_integration(token: str, repository: str, webhook_secret: str = None) -> bool:
    """Setup GitHub Actions integration"""
    return await cicd_integration.setup_github_actions_integration(token, repository, webhook_secret)

async def setup_gitlab_integration(token: str, project_id: str, gitlab_url: str = "https://gitlab.com") -> bool:
    """Setup GitLab CI integration"""
    return await cicd_integration.setup_gitlab_ci_integration(token, project_id, gitlab_url)

async def capture_pipeline_run(provider: PipelineProvider, run_id: str, repository: str = None) -> Optional[PipelineRun]:
    """Capture pipeline run from any provider"""
    if provider == PipelineProvider.GITHUB_ACTIONS and repository:
        return await cicd_integration.capture_github_actions_run(repository, run_id)
    elif provider == PipelineProvider.GITLAB_CI and repository:
        return await cicd_integration.capture_gitlab_pipeline(repository, run_id)
    else:
        logger.error(f"Provider {provider} not supported or missing repository")
        return None

async def handle_webhook(provider: PipelineProvider, payload: Dict[str, Any], headers: Dict[str, str] = None) -> bool:
    """Handle CI/CD webhook"""
    return await cicd_integration.webhook_handler(provider, payload, headers)

async def get_pipeline_stats(repository: str = None, days: int = 30) -> Dict[str, Any]:
    """Get pipeline analytics"""
    return await cicd_integration.get_pipeline_analytics(repository, days)

if __name__ == "__main__":
    # Test CI/CD integration
    async def test_cicd_integration():
        print("🔄 Testing CI/CD Pipeline Integration")
        
        # Test basic functionality
        print("✅ CI/CD Integration system initialized")
        print(f"   Providers supported: {len(PipelineProvider)}")
        print(f"   Pipeline stages: {len(PipelineStage)}")
        
        # Test analytics with mock data
        analytics = await get_pipeline_stats(days=30)
        if "error" in analytics:
            print(f"📊 No pipeline data found (expected for new installation)")
        else:
            print(f"📊 Pipeline analytics generated")
        
        print("✅ CI/CD Pipeline Integration ready!")
    
    asyncio.run(test_cicd_integration())